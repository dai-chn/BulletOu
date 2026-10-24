/// DropF3 (安全な打ち当たり) の訓練側列挙を書き出す照合ツール (shogi-nnue #105、2026-10-24)。
///
/// PackedSfenValue ファイルの先頭 N 件について `ShogiHalfKa2ThreatDropF3::map_features_split` を呼び、
/// DropF3 区間 (index >= HALFKA2_THREAT_TOTAL_DIMENSIONS) の index をオフセットを引いて 1 局面 1 行で書く:
///   <record>\t<stm の index をカンマ区切り (昇順)>\t<nstm の index>
/// shogi-nnue の tools/drop_f3_ref.py (cshogi の参照実装) と突合する (YO 側は nnuedump で同じ照合をする)。
/// CUDA 不要 (特徴の列挙だけ)。
///
/// Usage: cargo run --release --example dump_drop_f3 -- <psv_path> <out_path> [N]
use std::env;
use std::fs::File;
use std::io::{BufWriter, Read, Write};

use bulletou_lib::game::inputs::{HALFKA2_THREAT_TOTAL_DIMENSIONS, ShogiHalfKa2ThreatDropF3, SparseInputType};
use bulletou_lib::shogi::packed_sfen::PackedSfenValue;

const RECORD_SIZE: usize = 40;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: dump_drop_f3 <psv_path> <out_path> [N]");
        std::process::exit(2);
    }
    let n: usize = args.get(3).map(|s| s.parse().expect("N")).unwrap_or(20000);
    let mut file = File::open(&args[1]).expect("open psv");
    let mut out = BufWriter::new(File::create(&args[2]).expect("create out"));
    let inputs = ShogiHalfKa2ThreatDropF3;
    let mut buf = [0u8; RECORD_SIZE];
    let mut written = 0usize;
    let mut rows_stm = 0usize;
    let mut rows_nstm = 0usize;
    for i in 0..n {
        if file.read_exact(&mut buf).is_err() {
            break;
        }
        let psv: PackedSfenValue = unsafe { std::mem::transmute(buf) };
        let mut stm: Vec<usize> = Vec::new();
        let mut nstm: Vec<usize> = Vec::new();
        inputs.map_features_split(&psv, |s, ns| {
            if let Some(x) = s {
                if x >= HALFKA2_THREAT_TOTAL_DIMENSIONS {
                    stm.push(x - HALFKA2_THREAT_TOTAL_DIMENSIONS);
                }
            }
            if let Some(x) = ns {
                if x >= HALFKA2_THREAT_TOTAL_DIMENSIONS {
                    nstm.push(x - HALFKA2_THREAT_TOTAL_DIMENSIONS);
                }
            }
        });
        stm.sort_unstable();
        nstm.sort_unstable();
        rows_stm += stm.len();
        rows_nstm += nstm.len();
        let a: Vec<String> = stm.iter().map(|x| x.to_string()).collect();
        let b: Vec<String> = nstm.iter().map(|x| x.to_string()).collect();
        writeln!(out, "{}\t{}\t{}", i, a.join(","), b.join(",")).expect("write");
        written += 1;
    }
    out.flush().expect("flush");
    println!("dump_drop_f3: {written} records, rows stm {rows_stm} nstm {rows_nstm} -> {}", args[2]);
}
