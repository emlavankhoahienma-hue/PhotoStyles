import SwiftUI

@main
struct PhotoStylesApp: App {
    var body: some Scene {
        WindowGroup { PhotographicStylesDemoView().preferredColorScheme(.dark) }
    }
}
