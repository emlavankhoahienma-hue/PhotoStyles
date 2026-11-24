//
//  PhotographicStyles.swift
//  "Photographic Styles" cho iPhone XS → 18 Pro Max (iOS 15+, Swift 5 mode)
//
//  Kiến trúc (1 file, không thư viện ngoài):
//   • ColorMath / StyleKernel : biến đổi màu trong OKLab – tông, màu, undertone, bảo vệ da, nén gam màu Display P3
//   • LUTBuilder              : bake kernel thành 3D LUT (33³ cho preview, 65³ cho ảnh chụp), đa luồng
//   • StyleEngine             : tính LUT ở nền, gộp yêu cầu (coalesce) → kéo con trỏ không bao giờ giật hình
//   • StyleMotion             : lò xo tới hạn (critically damped) chạy trên CADisplayLink 60/120Hz + haptic
//   • StylePadView            : bảng 11×11 vẽ bằng Canvas, ô phình theo con trỏ, puck có bóng như Apple
//   • StyleCamera             : AVFoundation → Core Image (CIColorCubeWithColorSpace) → CAMetalLayer
//                               chụp xong áp LUT 65³ lên ảnh gốc, lưu HEIF 10-bit vào Photos
//
//  Info.plist cần:
//    NSCameraUsageDescription
//    NSPhotoLibraryAddUsageDescription
//    CADisableMinimumFrameDurationOnPhone = YES   (để animation chạy 120Hz trên ProMotion)
//
//  Dùng thử:  PhotographicStylesDemoView()
//  Gắn vào camera có sẵn:  StyleEngine + StyleCubeFilter (pipeline CIImage) + StyleController + StylePanel
//

import SwiftUI
import AVFoundation
import CoreImage
import ImageIO
import Metal
import Photos
import UIKit
import simd

// MARK: - Hằng số tinh chỉnh (đổi ở đây để "khớp Apple" theo mắt)

enum StyleTuning {
    static let tintMax = 0.045            // biên độ undertone tối đa (đơn vị OKLab a/b)
    static let toneContrast = 0.9         // |Tone| = 1  →  mũ đường cong S = 2^0.9
    static let saturationGain = 1.1       // Color = +1  →  ~+110% chroma (trước vibrance)
    static let skinProtect = 0.65         // vùng da chỉ nhận (1 - 0.65) mức tăng màu
    static let skinHue = 52.0 * Double.pi / 180
    static let swatchChroma = 0.115       // độ rực của ô màu trên bảng
    static let swatchL = 0.74             // độ sáng của ô màu trên bảng
}

// MARK: - Toán màu (OKLab, Display P3)

@inline(__always)
private func smooth(_ e0: Double, _ e1: Double, _ x: Double) -> Double {
    let t = min(max((x - e0) / (e1 - e0), 0), 1)
    return t * t * (3 - 2 * t)
}

private func cross3(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> SIMD3<Double> {
    SIMD3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

enum ColorMath {
    static let displayP3 = CGColorSpace(name: CGColorSpace.displayP3)!

    // Ma trận dựng bằng Double, nghịch đảo tính bằng số → round-trip chính xác, vùng trung tính không bị ám màu.
    private struct M3 {
        var r0: SIMD3<Double>, r1: SIMD3<Double>, r2: SIMD3<Double>
        func apply(_ v: SIMD3<Double>) -> SIMD3<Double> {
            SIMD3((r0 * v).sum(), (r1 * v).sum(), (r2 * v).sum())
        }
        func times(_ o: M3) -> M3 {
            let c0 = SIMD3(o.r0.x, o.r1.x, o.r2.x)
            let c1 = SIMD3(o.r0.y, o.r1.y, o.r2.y)
            let c2 = SIMD3(o.r0.z, o.r1.z, o.r2.z)
            return M3(r0: SIMD3((r0 * c0).sum(), (r0 * c1).sum(), (r0 * c2).sum()),
                      r1: SIMD3((r1 * c0).sum(), (r1 * c1).sum(), (r1 * c2).sum()),
                      r2: SIMD3((r2 * c0).sum(), (r2 * c1).sum(), (r2 * c2).sum()))
        }
        func inverse() -> M3 {
            let c0 = cross3(r1, r2), c1 = cross3(r2, r0), c2 = cross3(r0, r1)
            let det = (r0 * c0).sum()
            return M3(r0: SIMD3(c0.x, c1.x, c2.x) / det,
                      r1: SIMD3(c0.y, c1.y, c2.y) / det,
                      r2: SIMD3(c0.z, c1.z, c2.z) / det)
        }
    }

    private static let p3ToXYZ = M3(r0: SIMD3(0.4865709486, 0.2656676932, 0.1982172852),
                                    r1: SIMD3(0.2289745641, 0.6917385218, 0.0792869141),
                                    r2: SIMD3(0.0000000000, 0.0451133819, 1.0439443689))
    private static let xyzToLMS = M3(r0: SIMD3(0.8189330101, 0.3618667424, -0.1288597137),
                                     r1: SIMD3(0.0329845436, 0.9293118715, 0.0361456387),
                                     r2: SIMD3(0.0482003018, 0.2643662691, 0.6338517070))
    private static let lmsToLab = M3(r0: SIMD3(0.2104542553, 0.7936177850, -0.0040720468),
                                     r1: SIMD3(1.9779984951, -2.4285922050, 0.4505937099),
                                     r2: SIMD3(0.0259040371, 0.7827717662, -0.8086757660))
    private static let labToLMS = lmsToLab.inverse()
    /// RGB tuyến tính P3 → LMS, chuẩn hoá để trắng → (1,1,1) (neutral cho a = b = 0 tuyệt đối)
    private static let rgbToLMS: M3 = {
        let raw = xyzToLMS.times(p3ToXYZ)
        let w = raw.apply(SIMD3(1, 1, 1))
        return M3(r0: raw.r0 / w.x, r1: raw.r1 / w.y, r2: raw.r2 / w.z)
    }()
    private static let lmsToRGB = rgbToLMS.inverse()

    @inline(__always) private static func decode(_ v: Double) -> Double {
        v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4)
    }
    @inline(__always) private static func encode(_ v: Double) -> Double {
        let c = min(max(v, 0), 1)
        return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1 / 2.4) - 0.055
    }

    /// RGB gamma P3 (0…1) → OKLab (L, a, b)
    static func toOKLab(_ rgb: SIMD3<Double>) -> SIMD3<Double> {
        let lin = SIMD3(decode(rgb.x), decode(rgb.y), decode(rgb.z))
        let lms = rgbToLMS.apply(lin)
        return lmsToLab.apply(SIMD3(cbrt(lms.x), cbrt(lms.y), cbrt(lms.z)))
    }

    private static func linear(fromOKLab lab: SIMD3<Double>) -> SIMD3<Double> {
        let p = labToLMS.apply(lab)
        return lmsToRGB.apply(SIMD3(p.x * p.x * p.x, p.y * p.y * p.y, p.z * p.z * p.z))
    }

    private static func inGamut(_ lin: SIMD3<Double>) -> Bool {
        lin.min() >= -0.0005 && lin.max() <= 1.0005
    }

    /// OKLab → RGB gamma P3. Vượt gam màu thì nén chroma (giữ nguyên L và hue) thay vì cắt cụt kênh → không bệt màu.
    static func rgbFromOKLab(_ lab: SIMD3<Double>) -> SIMD3<Double> {
        let L = min(max(lab.x, 0), 1)
        var lin = linear(fromOKLab: SIMD3(L, lab.y, lab.z))
        if !inGamut(lin) {
            var lo = 0.0, hi = 1.0
            for _ in 0..<10 {
                let mid = (lo + hi) * 0.5
                if inGamut(linear(fromOKLab: SIMD3(L, lab.y * mid, lab.z * mid))) { lo = mid } else { hi = mid }
            }
            lin = linear(fromOKLab: SIMD3(L, lab.y * lo, lab.z * lo))
        }
        return SIMD3(encode(lin.x), encode(lin.y), encode(lin.z))
    }
}

// MARK: - Tham số style

struct StyleParams: Equatable {
    var padX = 0.0, padY = 0.0          // -1…1 : trục a (xanh lá ←→ hồng) và trục b (xanh dương ↓↑ hổ phách)
    var tone = 0.0, color = 0.0         // -1…1
    var blackLift = 0.0                 // nâng điểm đen (0…0.15)
    var mono = 0.0                      // 0 = màu, 1 = đen trắng
    var isIdentity: Bool {
        abs(padX) + abs(padY) + abs(tone) + abs(color) + blackLift + mono < 2e-3
    }
}

/// Hàm biến đổi 1 màu. Chạy trên lưới LUT nên chi phí không phụ thuộc độ phân giải ảnh.
struct StyleKernel {
    private let p: StyleParams
    private let g: Double
    private let lift: Double
    private let tintA: Double
    private let tintB: Double

    init(_ p: StyleParams) {
        self.p = p
        g = pow(2, StyleTuning.toneContrast * p.tone)
        lift = p.blackLift + max(0, -p.tone) * 0.03
        tintA = p.padX * StyleTuning.tintMax
        tintB = p.padY * StyleTuning.tintMax
    }

    func apply(_ rgb: SIMD3<Double>) -> SIMD3<Double> {
        let lab = ColorMath.toOKLab(rgb)

        // 1) Tông: nâng đen + đường cong chữ S quanh điểm giữa (đầu mút cố định → không cháy sáng)
        let L = curve(lift + (1 - lift) * lab.x)

        // 2) Màu: vibrance (tăng nhiều ở vùng nhạt) + bảo vệ tông da
        var a = lab.y, b = lab.z
        let C = (a * a + b * b).squareRoot()
        let k0: Double
        if p.color >= 0 {
            let vib = 1 - 0.6 * smooth(0.04, 0.20, C)
            let skin = Self.skinWeight(atan2(b, a), C)
            k0 = 1 + p.color * StyleTuning.saturationGain * vib * (1 - StyleTuning.skinProtect * skin)
        } else {
            k0 = 1 + p.color
        }
        let k = k0 * (1 - p.mono)
        a *= k; b *= k

        // 3) Undertone từ bảng màu: mạnh ở vùng trung tính, yếu dần về đen/trắng
        let w = 0.3 + 0.7 * (4 * L * (1 - L))
        let gain = 1 - 0.5 * p.mono
        a += tintA * w * gain
        b += tintB * w * gain

        return ColorMath.rgbFromOKLab(SIMD3(L, a, b))
    }

    private func curve(_ x: Double) -> Double {
        let pv = 0.55
        let c = min(max(x, 0), 1)
        return c < pv ? pv * pow(c / pv, g) : 1 - (1 - pv) * pow((1 - c) / (1 - pv), g)
    }

    static func skinWeight(_ h: Double, _ C: Double) -> Double {
        var d = h - StyleTuning.skinHue
        if d > .pi { d -= 2 * .pi } else if d < -.pi { d += 2 * .pi }
        let hue = 1 - smooth(0.30, 0.65, abs(d))
        let chroma = smooth(0.015, 0.045, C) * (1 - smooth(0.13, 0.20, C))
        return hue * chroma
    }
}

// MARK: - Bake LUT 3D

enum LUTBuilder {
    /// Trả về dữ liệu cho CIColorCubeWithColorSpace: RGBA float32, đã premultiply (alpha = 1), R chạy nhanh nhất.
    static func build(_ p: StyleParams, dimension n: Int) -> Data {
        var cube = [Float](repeating: 1, count: n * n * n * 4)
        let kernel = StyleKernel(p)
        let identity = p.isIdentity
        let inv = 1.0 / Double(n - 1)
        cube.withUnsafeMutableBufferPointer { buf in
            let base = buf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: n) { bi in
                for gi in 0..<n {
                    for ri in 0..<n {
                        let src = SIMD3<Double>(Double(ri) * inv, Double(gi) * inv, Double(bi) * inv)
                        let out = identity ? src : kernel.apply(src)
                        let o = ((bi * n + gi) * n + ri) * 4
                        base[o] = Float(out.x)
                        base[o + 1] = Float(out.y)
                        base[o + 2] = Float(out.z)
                        base[o + 3] = 1
                    }
                }
            }
        }
        return cube.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}

// MARK: - Engine: tính LUT nền, gộp yêu cầu

final class StyleEngine {
    static let previewDimension = 33      // máy yếu (XS/XR) có thể hạ xuống 25
    static let stillDimension = 65

    private let lock = NSLock()
    private let queue = DispatchQueue(label: "style.lut", qos: .userInteractive)
    private var wanted = StyleParams()
    private var stamp = 0
    private var computing = false
    private var ready: (data: Data?, stamp: Int) = (nil, 0)   // data = nil → identity, bỏ qua filter

    /// Gọi thoải mái mỗi khung hình khi kéo: chỉ bản mới nhất được tính.
    func update(_ p: StyleParams) {
        lock.lock()
        wanted = p
        stamp += 1
        let shouldStart = !computing
        computing = true
        lock.unlock()
        if shouldStart { queue.async { [weak self] in self?.pump() } }
    }

    private func pump() {
        while true {
            lock.lock(); let p = wanted; let s = stamp; lock.unlock()
            let data: Data? = p.isIdentity ? nil : LUTBuilder.build(p, dimension: Self.previewDimension)
            lock.lock()
            ready = (data, s)
            let finished = (s == stamp)
            if finished { computing = false }
            lock.unlock()
            if finished { return }
        }
    }

    func snapshot() -> (data: Data?, stamp: Int) {
        lock.lock(); defer { lock.unlock() }
        return ready
    }

    var currentParams: StyleParams {
        lock.lock(); defer { lock.unlock() }
        return wanted
    }
}

/// Bộ lọc dùng lại giữa các khung hình (mỗi queue render giữ 1 instance – CIFilter không thread-safe).
/// Dùng cái này để gắn style vào pipeline CIImage có sẵn của app.
final class StyleCubeFilter {
    private let filter = CIFilter(name: "CIColorCubeWithColorSpace")!
    private var lastStamp = -1
    private var active = false

    init() {
        filter.setValue(StyleEngine.previewDimension, forKey: "inputCubeDimension")
        filter.setValue(ColorMath.displayP3, forKey: "inputColorSpace")
    }

    func apply(_ image: CIImage, engine: StyleEngine) -> CIImage {
        let snap = engine.snapshot()
        if snap.stamp != lastStamp {
            lastStamp = snap.stamp
            active = snap.data != nil
            if let d = snap.data { filter.setValue(d, forKey: "inputCubeData") }
        }
        guard active else { return image }
        filter.setValue(image, forKey: kCIInputImageKey)
        return filter.outputImage ?? image
    }
}

// MARK: - Chuyển động: lò xo tới hạn trên CADisplayLink

struct Spring2 {
    var value = SIMD2<Double>(0, 0)
    var velocity = SIMD2<Double>(0, 0)
    var target = SIMD2<Double>(0, 0)

    /// Nghiệm giải tích của lò xo tới hạn → ổn định vô điều kiện ở mọi dt (60/120Hz, rớt khung vẫn mượt).
    mutating func step(_ dt: Double, omega w: Double) {
        let e = exp(-w * dt)
        let x = value - target
        let t = velocity + w * x
        value = (x + t * dt) * e + target
        velocity = (velocity - w * dt * t) * e
    }
    var isSettled: Bool {
        let d = value - target
        return (d * d).sum() < 1e-8 && (velocity * velocity).sum() < 1e-6
    }
    mutating func snap() {
        value = target
        velocity = SIMD2<Double>(0, 0)
    }
}

@MainActor
private final class DisplayLinkProxy: NSObject {
    weak var owner: StyleMotion?
    init(_ owner: StyleMotion) { self.owner = owner }
    @objc func tick(_ link: CADisplayLink) {
        guard let owner = owner else { link.invalidate(); return }
        owner.step(link)
    }
}

@MainActor
final class StyleMotion: ObservableObject {
    static let n = 11                                             // bảng 11 × 11

    @Published private(set) var position = SIMD2<Double>(0, 0)    // vị trí puck đã làm mượt, -1…1
    @Published private(set) var touch = 0.0                       // 0…1: mức "đang chạm" (puck to lên, ô phình)
    /// (pad, tone/color, blackLift/mono) – đều là giá trị đã làm mượt
    var onChange: ((SIMD2<Double>, SIMD2<Double>, SIMD2<Double>) -> Void)?
    var stiffness = 30.0                                          // ω: lớn = bám ngón nhanh, nhỏ = trượt "lười"

    private var pad = Spring2()
    private var toneColor = Spring2()
    private var extras = Spring2()
    private var touchSpring = Spring2()
    private var isDown = false
    private var lastCell = SIMD2<Int>(5, 5)
    private var link: CADisplayLink?
    private let selection = UISelectionFeedbackGenerator()
    private let impact = UIImpactFeedbackGenerator(style: .rigid)

    // --- Cử chỉ ---
    func touchBegan() {
        guard !isDown else { return }
        isDown = true
        touchSpring.target = SIMD2(1, 0)
        selection.prepare(); impact.prepare()
        wake()
    }

    func touchMoved(to p: SIMD2<Double>) {
        var q = p
        if abs(q.x) < 0.04 && abs(q.y) < 0.04 { q = SIMD2(0, 0) }
        let cell = Self.cellIndex(q)
        if cell != lastCell {
            lastCell = cell
            if cell == SIMD2(5, 5) { impact.impactOccurred(intensity: 0.9) } else { selection.selectionChanged() }
            selection.prepare()
        }
        pad.target = q
        wake()
    }

    func touchEnded() {
        isDown = false
        touchSpring.target = SIMD2(0, 0)
        wake()
    }

    // --- Điều khiển bằng code (chọn preset, slider) ---
    func glide(to p: SIMD2<Double>) {
        pad.target = p
        lastCell = Self.cellIndex(p)
        wake()
    }
    func setToneColor(_ tone: Double, _ color: Double) {
        toneColor.target = SIMD2(tone, color)
        wake()
    }
    func setExtras(blackLift: Double, mono: Double) {
        extras.target = SIMD2(blackLift, mono)
        wake()
    }

    static func cellIndex(_ p: SIMD2<Double>) -> SIMD2<Int> {
        let m = Double(n - 1)
        return SIMD2(Int(((p.x + 1) / 2 * m).rounded()), Int(((1 - p.y) / 2 * m).rounded()))
    }

    // --- Vòng lặp ---
    private func wake() {
        guard link == nil else { return }
        let l = CADisplayLink(target: DisplayLinkProxy(self), selector: #selector(DisplayLinkProxy.tick(_:)))
        l.preferredFrameRateRange = CAFrameRateRange(minimum: 60, maximum: 120, preferred: 120)
        l.add(to: .main, forMode: .common)
        link = l
    }

    fileprivate func step(_ l: CADisplayLink) {
        let dt = min(max(l.targetTimestamp - l.timestamp, 1.0 / 240), 1.0 / 30)
        pad.step(dt, omega: stiffness)
        toneColor.step(dt, omega: 38)
        extras.step(dt, omega: 22)
        touchSpring.step(dt, omega: 26)
        if pad.isSettled && toneColor.isSettled && extras.isSettled && touchSpring.isSettled {
            pad.snap(); toneColor.snap(); extras.snap(); touchSpring.snap()
            link?.invalidate()
            link = nil
        }
        position = pad.value
        touch = touchSpring.value.x
        onChange?(pad.value, toneColor.value, extras.value)
    }
}

// MARK: - Preset (giá trị khởi điểm, chỉnh theo ý)

enum StylePreset: String, CaseIterable, Identifiable {
    case standard, richContrast, vibrant, warm, cool, cozy, dramatic, quiet, ethereal, mutedBW, starkBW

    var id: String { rawValue }

    var title: String {
        switch self {
        case .standard: return "Tiêu chuẩn"
        case .richContrast: return "Tương phản"
        case .vibrant: return "Rực rỡ"
        case .warm: return "Ấm"
        case .cool: return "Lạnh"
        case .cozy: return "Ấm cúng"
        case .dramatic: return "Kịch tính"
        case .quiet: return "Dịu"
        case .ethereal: return "Mơ màng"
        case .mutedBW: return "Đen trắng dịu"
        case .starkBW: return "Đen trắng gắt"
        }
    }

    struct Look {
        var tone = 0.0, color = 0.0, padX = 0.0, padY = 0.0, blackLift = 0.0, mono = false
    }

    var look: Look {
        switch self {
        case .standard: return Look()
        case .richContrast: return Look(tone: 0.45, color: 0.25)
        case .vibrant: return Look(tone: 0.10, color: 0.65)
        case .warm: return Look(tone: 0.05, color: 0.10, padX: 0.20, padY: 0.55)
        case .cool: return Look(tone: 0.05, color: 0.05, padX: -0.20, padY: -0.55)
        case .cozy: return Look(tone: -0.10, color: 0.00, padX: 0.35, padY: 0.35, blackLift: 0.02)
        case .dramatic: return Look(tone: 0.75, color: -0.15, padX: 0.0, padY: -0.15)
        case .quiet: return Look(tone: -0.30, color: -0.40, padX: -0.10, padY: 0.0, blackLift: 0.04)
        case .ethereal: return Look(tone: -0.55, color: -0.15, padX: -0.10, padY: -0.20, blackLift: 0.06)
        case .mutedBW: return Look(tone: -0.20, blackLift: 0.03, mono: true)
        case .starkBW: return Look(tone: 0.85, mono: true)
        }
    }
}

// MARK: - Controller

@MainActor
final class StyleController: ObservableObject {
    let motion = StyleMotion()
    @Published private(set) var preset: StylePreset = .standard
    @Published var tone = 0.0 { didSet { motion.setToneColor(tone, color) } }
    @Published var color = 0.0 { didSet { motion.setToneColor(tone, color) } }
    private let engine: StyleEngine

    init(engine: StyleEngine) {
        self.engine = engine
        motion.onChange = { [weak self] pad, tc, extra in
            self?.engine.update(StyleParams(padX: pad.x, padY: pad.y,
                                            tone: tc.x, color: tc.y,
                                            blackLift: extra.x, mono: extra.y))
        }
    }

    func select(_ p: StylePreset) {
        preset = p
        let l = p.look
        withAnimation(.spring(response: 0.4, dampingFraction: 0.86)) {
            tone = l.tone
            color = l.color
        }
        motion.glide(to: SIMD2(l.padX, l.padY))
        motion.setExtras(blackLift: l.blackLift, mono: l.mono ? 1 : 0)
    }

    func reset() { select(.standard) }
}

// MARK: - Bảng màu 11×11

enum StylePadColors {
    static let n = StyleMotion.n

    /// Màu xem trước của một điểm trên bảng: (u, v) ∈ [-1, 1]² → a/b của OKLab, tâm = xám trung tính.
    static func color(u: Double, v: Double) -> Color {
        let c = ColorMath.rgbFromOKLab(SIMD3(StyleTuning.swatchL, u * StyleTuning.swatchChroma, v * StyleTuning.swatchChroma))
        return Color(.displayP3, red: c.x, green: c.y, blue: c.z, opacity: 1)
    }

    static let cells: [Color] = (0..<(n * n)).map { idx in
        let i = idx % n, j = idx / n
        return color(u: Double(i) / Double(n - 1) * 2 - 1, v: 1 - Double(j) / Double(n - 1) * 2)
    }
}

struct StylePadView: View {
    @ObservedObject var model: StyleMotion

    var body: some View {
        GeometryReader { geo in
            let side = min(geo.size.width, geo.size.height)
            let n = StyleMotion.n
            let cell = side / CGFloat(n)
            let pc = Self.point(model.position, side: side, cell: cell)
            let puck = cell * CGFloat(1.7 + 0.5 * model.touch)

            ZStack {
                Canvas { ctx, _ in
                    for j in 0..<n {
                        for i in 0..<n {
                            let cx = (CGFloat(i) + 0.5) * cell
                            let cy = (CGFloat(j) + 0.5) * cell
                            let d = Double(hypot(cx - pc.x, cy - pc.y) / cell)
                            // ô gần con trỏ phình ra (Gaussian), mạnh hơn khi đang chạm
                            let near = exp(-d * d / 2.4) * (0.3 + 0.7 * model.touch)
                            let s = cell * CGFloat(0.5 + 0.42 * near)
                            let r = CGRect(x: cx - s / 2, y: cy - s / 2, width: s, height: s)
                            ctx.fill(Path(roundedRect: r, cornerRadius: s * 0.34, style: .continuous),
                                      with: .color(StylePadColors.cells[j * n + i]))
                        }
                    }
                }
                Circle()
                    .fill(StylePadColors.color(u: model.position.x, v: model.position.y))
                    .overlay(Circle().strokeBorder(Color.white, lineWidth: 3.5))
                    .frame(width: puck, height: puck)
                    .shadow(color: Color.black.opacity(0.35), radius: 8, x: 0, y: 3)
                    .position(pc)
                    .allowsHitTesting(false)
            }
            .frame(width: side, height: side)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { v in
                        model.touchBegan()
                        model.touchMoved(to: Self.unit(v.location, side: side, cell: cell))
                    }
                    .onEnded { _ in model.touchEnded() }
            )
            .position(x: geo.size.width / 2, y: geo.size.height / 2)
        }
        .aspectRatio(1, contentMode: .fit)
    }

    /// u,v ∈ [-1,1] → toạ độ điểm ảnh. Tâm các ô ngoài cùng ứng với ±1, ô giữa ứng với 0.
    static func point(_ p: SIMD2<Double>, side: CGFloat, cell: CGFloat) -> CGPoint {
        let span = side - cell
        return CGPoint(x: cell / 2 + CGFloat((p.x + 1) / 2) * span,
                       y: cell / 2 + CGFloat((1 - p.y) / 2) * span)
    }

    static func unit(_ l: CGPoint, side: CGFloat, cell: CGFloat) -> SIMD2<Double> {
        let span = Double(side - cell)
        let x = (Double(l.x - cell / 2) / span) * 2 - 1
        let y = 1 - (Double(l.y - cell / 2) / span) * 2
        return SIMD2(min(max(x, -1), 1), min(max(y, -1), 1))
    }
}

// MARK: - Panel: preset + bảng màu + Tông / Màu

struct StylePanel: View {
    @ObservedObject var style: StyleController

    var body: some View {
        VStack(spacing: 16) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(StylePreset.allCases) { p in
                        Button { style.select(p) } label: {
                            Text(p.title)
                                .font(.footnote.weight(.semibold))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 7)
                                .foregroundColor(style.preset == p ? .black : .white)
                                .background(style.preset == p ? Color.white : Color.white.opacity(0.16), in: Capsule())
                        }
                    }
                }
                .padding(.horizontal, 2)
            }

            StylePadView(model: style.motion)
                .frame(maxWidth: 320)

            VStack(spacing: 6) {
                row("Tông", value: $style.tone)
                row("Màu", value: $style.color)
            }

            Button("Đặt lại") { style.reset() }
                .font(.footnote.weight(.medium))
                .foregroundColor(Color.white.opacity(0.8))
        }
        .environment(\.colorScheme, .dark)
    }

    private func row(_ title: String, value: Binding<Double>) -> some View {
        HStack(spacing: 10) {
            Text(title).font(.footnote).frame(width: 36, alignment: .leading)
            Slider(value: Binding(get: { value.wrappedValue },
                                  set: { value.wrappedValue = abs($0) < 0.03 ? 0 : $0 }),   // detent ở 0
                    in: -1...1)
            Text("\(Int((value.wrappedValue * 100).rounded()))")
                .font(.footnote.monospacedDigit())
                .frame(width: 36, alignment: .trailing)
        }
        .foregroundColor(.white)
    }
}

// MARK: - Preview Metal (Core Image → CAMetalLayer)

final class PreviewRenderer {
    let metalLayer: CAMetalLayer
    private let engine: StyleEngine
    private let queue: MTLCommandQueue
    private let ciContext: CIContext
    private let cube = StyleCubeFilter()

    init(layer: CAMetalLayer, engine: StyleEngine) {
        self.engine = engine
        let device = MTLCreateSystemDefaultDevice()!
        let q = device.makeCommandQueue()!
        queue = q
        ciContext = CIContext(mtlCommandQueue: q, options: [.cacheIntermediates: false])
        metalLayer = layer
        layer.device = device
        layer.pixelFormat = .bgra8Unorm
        layer.framebufferOnly = false
        layer.colorspace = ColorMath.displayP3
    }

    /// Gọi từ queue camera cho từng khung hình.
    func render(_ buffer: CVPixelBuffer) {
        autoreleasepool {
            let size = metalLayer.drawableSize
            guard size.width > 0, size.height > 0,
                  let drawable = metalLayer.nextDrawable(),
                  let cb = queue.makeCommandBuffer() else { return }

            var image = cube.apply(CIImage(cvPixelBuffer: buffer), engine: engine)

            // aspect-fill vào drawable
            let e = image.extent
            let s = max(size.width / e.width, size.height / e.height)
            let w = e.width * s, h = e.height * s
            let t = CGAffineTransform(translationX: -e.minX, y: -e.minY)
                .concatenating(CGAffineTransform(scaleX: s, y: s))
                .concatenating(CGAffineTransform(translationX: (size.width - w) / 2, y: (size.height - h) / 2))
            image = image.transformed(by: t)

            ciContext.render(image, to: drawable.texture, commandBuffer: cb,
                             bounds: CGRect(origin: .zero, size: size), colorSpace: ColorMath.displayP3)
            cb.present(drawable)
            cb.commit()
        }
    }
}

final class MetalPreviewView: UIView {
    override class var layerClass: AnyClass { CAMetalLayer.self }
    private(set) var renderer: PreviewRenderer!

    init(engine: StyleEngine) {
        super.init(frame: .zero)
        backgroundColor = .black
        renderer = PreviewRenderer(layer: layer as! CAMetalLayer, engine: engine)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func layoutSubviews() {
        super.layoutSubviews()
        let s = traitCollection.displayScale > 0 ? traitCollection.displayScale : 3
        renderer.metalLayer.contentsScale = s
        renderer.metalLayer.drawableSize = CGSize(width: bounds.width * s, height: bounds.height * s)
    }
}

struct StyledPreview: UIViewRepresentable {
    let camera: StyleCamera
    func makeUIView(context: Context) -> MetalPreviewView {
        let v = MetalPreviewView(engine: camera.engine)
        camera.preview = v.renderer
        return v
    }
    func updateUIView(_ uiView: MetalPreviewView, context: Context) {}
}

// MARK: - Camera

final class StyleCamera: NSObject, ObservableObject {
    let engine = StyleEngine()
    weak var preview: PreviewRenderer?

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let photoOutput = AVCapturePhotoOutput()
    private let sessionQueue = DispatchQueue(label: "style.session")
    private let videoQueue = DispatchQueue(label: "style.video", qos: .userInitiated)
    private let processQueue = DispatchQueue(label: "style.process", qos: .userInitiated)
    private static let stillContext = CIContext(options: [.cacheIntermediates: false])
    private var configured = false
    private var pending: [Int64: StyleParams] = [:]
    private let pendingLock = NSLock()

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
            guard ok, let self = self else { return }
            self.sessionQueue.async {
                self.configure()
                if !self.session.isRunning { self.session.startRunning() }
            }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            if self?.session.isRunning == true { self?.session.stopRunning() }
        }
    }

    private static func portrait(_ c: AVCaptureConnection) {
        if #available(iOS 17.0, *) {
            if c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
        } else if c.isVideoOrientationSupported {
            c.videoOrientation = .portrait
        }
    }

    private func configure() {
        guard !configured else { return }
        session.beginConfiguration()
        session.sessionPreset = .photo
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }
        if session.canAddOutput(photoOutput) {
            session.addOutput(photoOutput)
            photoOutput.maxPhotoQualityPrioritization = .quality
        }
        if let c = videoOutput.connection(with: .video) { Self.portrait(c) }
        session.commitConfiguration()
        configured = true
    }

    func capture() {
        let params = engine.currentParams
        sessionQueue.async { [weak self] in
            guard let self = self, self.session.isRunning else { return }
            let settings = self.photoOutput.availablePhotoCodecTypes.contains(.hevc)
                ? AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.hevc])
                : AVCapturePhotoSettings()
            settings.photoQualityPrioritization = .quality
            if let c = self.photoOutput.connection(with: .video) { Self.portrait(c) }
            self.pendingLock.lock()
            self.pending[settings.uniqueID] = params
            self.pendingLock.unlock()
            self.photoOutput.capturePhoto(with: settings, delegate: self)
        }
    }
}
