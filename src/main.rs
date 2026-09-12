//! `go2-run` — Unitree Go2 を動かす実行ファイル。
//!
//! **モデルベースとRLベースの両方を 1 本で持つ:**
//!
//! ```text
//! go2-run check|dump|sim ... --robot robots/go2.toml
//!                                misa-runner のサブコマンドそのまま
//!                                （歩容 = quadruped-gait、WBC、MuJoCo sim）
//! GO2_IFACE=eth0 go2-run run --robot robots/go2.toml --pilot keys
//!                                歩容/WBC を実機で（Go2Backend が rt/lowcmd を話す）
//! GO2_IFACE=eth0 go2-run policy --model exported/policy.onnx
//!                                MIT モード RL 方策（misa-policy-runner、
//!                                50 Hz 推論 + 500 Hz MIT + 支持脚 τ_ff）
//! ```
//!
//! `policy` だけはここで受ける — misa-runner の run ループには RL の二層
//! 構造（50 Hz 方策 / 500 Hz 低レベル + 毎周期の τ_ff）の差し込み口が
//! まだ無い。それ以外は [`misa_runner::main_with`] に Go2Backend を差して
//! 丸投げする（namiashi-runner2 / keel-runner と同じ形）。

mod backend;
mod estimator;
mod go2_plant;
mod policy;
#[cfg(feature = "sim")]
mod policy_sim;

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(|s| s.as_str()) == Some("policy") {
        env_logger::Builder::from_env(
            env_logger::Env::default().default_filter_or("info"),
        )
        .format_timestamp_millis()
        .init();
        return match policy::run(&args[1..]) {
            Ok(()) => std::process::ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("エラー: {e}");
                std::process::ExitCode::FAILURE
            }
        };
    }
    misa_runner::main_with(&[&backend::Go2Backend])
}
