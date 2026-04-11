use std::sync::Once;

static BASELINE_ONCE: Once = Once::new();

pub fn print_baseline_manifest(lines: &[&str]) {
    BASELINE_ONCE.call_once(|| {
        println!("BASELINE_MANIFEST_BEGIN");
        for line in lines {
            println!("{line}");
        }
        println!("BASELINE_MANIFEST_END");
    });
}
