//! Window construction for the app's three windows:
//!   main    — the bundled setup flow (first run only)
//!   brain   — the user's remote dashboard, wrapped (every run after setup)
//!   details — the local "Connection details" panel
//!
//! The `brain` window is remote content: it gets NO Tauri IPC (it isn't listed
//! in any capability). The only thing injected is the dashboard's own
//! localStorage auth keys, guarded so they're set solely on the user's own
//! Worker origin.

use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindowBuilder};

pub fn open_setup_window(app: &AppHandle) -> tauri::Result<()> {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.set_focus();
        return Ok(());
    }
    WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
        .title("Second Brain")
        .inner_size(940.0, 700.0)
        .min_inner_size(760.0, 560.0)
        .build()?;
    Ok(())
}

pub fn open_wrapper_window(
    app: &AppHandle,
    worker_url: &str,
    auth_token: &str,
) -> tauri::Result<()> {
    if let Some(w) = app.get_webview_window("brain") {
        let _ = w.set_focus();
        return Ok(());
    }
    let origin = worker_url.trim_end_matches('/');
    // serde_json turns the values into safely-escaped JS string literals.
    let origin_js = serde_json::to_string(origin).expect("string serializes");
    let token_js = serde_json::to_string(auth_token).expect("string serializes");
    let init = format!(
        r#"(function () {{
  try {{
    if (location.origin === {origin_js}) {{
      localStorage.setItem('sb_url', {origin_js});
      localStorage.setItem('sb_token', {token_js});
    }}
  }} catch (_) {{}}
}})();"#
    );
    let url: tauri::Url = format!("{origin}/")
        .parse()
        .map_err(|_| tauri::Error::WindowNotFound)?;
    WebviewWindowBuilder::new(app, "brain", WebviewUrl::External(url))
        .title("Second Brain")
        .inner_size(1180.0, 820.0)
        .min_inner_size(720.0, 480.0)
        .initialization_script(&init)
        .build()?;
    Ok(())
}

pub fn open_details_window(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("details") {
        let _ = w.set_focus();
        return;
    }
    let _ = WebviewWindowBuilder::new(app, "details", WebviewUrl::App("details.html".into()))
        .title("Connection details")
        .inner_size(540.0, 700.0)
        .min_inner_size(460.0, 520.0)
        .build();
}
