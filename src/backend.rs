//! Go2Backend — misa-runner の `run` / `stance` / `bridge` に Go2 を繋ぐ。
//!
//! misa-runner が Backend を当たるのは **プロファイルが `kind = "ros2"` の
//! ときだけ**（kind = "serial" は組み込みの SerialPlant に直結）。Go2 の
//! rt/lowcmd / rt/lowstate は CycloneDDS のトピックなので、この分類は実態
//! とも合っている。`robots/go2.toml` は kind = "ros2" とし、`mit_gains` を
//! Position 制御の kp/kd に使う。
//!
//! NIC は環境変数 **`GO2_IFACE`**（例 `GO2_IFACE=eth0 go2-run run ...`）。
//! 未指定なら既定経路。プロファイルには書けない — `Ros2Hardware` に NIC の
//! 置き場が無く、別の意味のフィールドに間借りさせると事故のもとになる。
//!
//! sport_mode は接続時に自動で解除する（低レベル指令と同居できない）。
//! `GO2_NO_RELEASE=1` で抑止（別の手段で解除済みのとき用）。

use misa_runner::misa_core::{
    GaitSelect, Intent, ModeRequest, Pilot, Time,
};
use misa_runner::{AppConfig, Backend};

use crate::go2_plant::Go2Plant;

pub struct Go2Backend;

/// 速度 0 の Stand を返し続ける操縦者。**実機の操縦には `--pilot keys` を
/// 重ねる**（misa-runner が接続後に差し替える）。受信機は無いので、これが
/// 無操縦時の安全側の既定になる。
struct HoldPilot;

impl Pilot for HoldPilot {
    fn poll(&mut self, now: Time) -> Intent {
        Intent {
            time: now,
            mode: ModeRequest::Stand,
            gait: GaitSelect::Crawl,
            link_ok: true,
            ..Intent::default()
        }
    }

    fn status_line(&self) -> String {
        "HoldPilot（速度 0 保持。操縦は --pilot keys）".into()
    }
}

/// 接続に使う NIC 名。
pub fn iface_from_env() -> Option<String> {
    std::env::var("GO2_IFACE").ok().filter(|s| !s.is_empty())
}

/// sport_mode を解除する（`GO2_NO_RELEASE=1` で抑止）。
pub fn release_sport_mode(iface: Option<&str>) -> Result<(), String> {
    if std::env::var("GO2_NO_RELEASE").ok().as_deref() == Some("1") {
        log::info!("GO2_NO_RELEASE=1: sport_mode の解除を省きます");
        return Ok(());
    }
    let sw = unitree_rpc::MotionSwitcher::new(iface.unwrap_or(""))
        .map_err(|e| format!("MotionSwitcher: {e}"))?;
    sw.release().map_err(|e| format!("sport_mode release: {e}"))?;
    log::info!("sport_mode を解除しました");
    Ok(())
}

impl Backend for Go2Backend {
    fn connect(
        &self,
        cfg: &AppConfig,
    ) -> Option<
        Result<
            (
                Box<dyn misa_runner::misa_core::Plant>,
                Box<dyn misa_runner::misa_core::Pilot>,
            ),
            String,
        >,
    > {
        // kind = "ros2" のときしか呼ばれないが、将来別のブリッジ機が並んだ
        // ときのために名前でも絞る。
        if cfg.name != "go2" {
            return None;
        }
        let gains = cfg.hardware.mit_gains()?; // ros2 なら必ず Some
        Some((|| {
            let iface = iface_from_env();
            release_sport_mode(iface.as_deref())?;
            let plant = Go2Plant::connect(iface.as_deref(), gains.kp, gains.kd)?;
            Ok((
                Box::new(plant) as Box<dyn misa_runner::misa_core::Plant>,
                Box::new(HoldPilot) as Box<dyn misa_runner::misa_core::Pilot>,
            ))
        })())
    }

    /// `bridge`: 指令は出さず、LowState の到達と周期だけ確かめる。
    fn diagnose(&self, cfg: &AppConfig, secs: Option<f64>) -> Option<Result<(), String>> {
        if cfg.name != "go2" {
            return None;
        }
        Some((|| {
            let iface = iface_from_env();
            // 指令は脱力のまま = 何も送らない。sport_mode も触らない。
            let gains = cfg
                .hardware
                .mit_gains()
                .ok_or("kind = \"ros2\" のプロファイルではありません")?;
            let mut plant = Go2Plant::connect(iface.as_deref(), gains.kp, gains.kd)?;
            let secs = secs.unwrap_or(10.0);
            let t0 = std::time::Instant::now();
            let mut n = 0u64;
            while t0.elapsed().as_secs_f64() < secs {
                // 受信だけ回す（exchange は指令を送るので使わない）
                if plant.poll_state()? {
                    n += 1;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            println!(
                "bridge: {secs:.0} 秒で LowState 新着 {} 回（~{:.0} Hz 相当）",
                n,
                n as f64 / secs
            );
            Ok(())
        })())
    }
}
