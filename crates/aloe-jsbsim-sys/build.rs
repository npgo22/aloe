use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing manifest dir"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("failed to locate workspace root")
        .to_path_buf();
    let jsbsim_dir = workspace_root.join("third_party/jsbsim");
    let wrapper_dir = manifest_dir.join("wrapper");

    println!("cargo:rerun-if-changed={}", wrapper_dir.display());
    println!(
        "cargo:rerun-if-changed={}",
        jsbsim_dir.join("src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        workspace_root.join(".gitmodules").display()
    );

    let dst = cmake::Config::new(&jsbsim_dir)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_DOCS", "OFF")
        .define("BUILD_PYTHON_MODULE", "OFF")
        .define("BUILD_MATLAB_SFUNCTION", "OFF")
        .define("BUILD_JULIA_PACKAGE", "OFF")
        .profile("Release")
        .build();

    let include_dir = dst.join("include");
    let lib_dir = dst.join("lib");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include(&include_dir)
        .include(jsbsim_dir.join("src"))
        .include(&wrapper_dir)
        .file(wrapper_dir.join("aloe_jsbsim_wrapper.cpp"));

    build.flag_if_supported("-Wno-unused-parameter");
    build.flag_if_supported("/wd4100");
    build.compile("aloe_jsbsim_wrapper");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=JSBSim");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=m");
}
