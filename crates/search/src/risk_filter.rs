//! `search risk-filter`: the stored trick-3 rows that a G2 model (`nn::leafrisk`, "where O_3 is
//! wrong") flags as risky, for the specialist network S (docs/m3-hybrid/specialist.md).
//!
//! Each row of a stored label shard is decoded back into its position (`nn::encode::decode`),
//! scored by G2 with the network's own value there (`Oracle::eval`), and kept when its score is
//! at or above the model's budget threshold for the chosen step (`threshold_for_share`): the same
//! cut the play rule uses to send a frontier position to the exact solver. Rows are kept in
//! their shard order with their labels unchanged; nothing is re-weighted.
//!
//! The scores are whole numbers from the model's trees, so the selection is the same on any thread
//! count: the rows are cut into contiguous chunks and the results are joined in order.

use nn::shard::Row;

/// The G2 score of every row of `rows` that is a trick-3 start (`None` for the others), computed
/// by `score` on `threads` threads. The result is in row order and does not depend on `threads`.
///
/// # Errors
/// The first row whose features are not an encoding (`score` returns `Err`).
pub fn score_rows<F>(rows: &[Row], threads: usize, score: F) -> Result<Vec<Option<i64>>, String>
where
    F: Fn(&Row) -> Result<Option<i64>, String> + Sync,
{
    let threads = threads.clamp(1, rows.len().max(1));
    let chunk = rows.len().div_ceil(threads).max(1);
    let parts: Vec<Result<Vec<Option<i64>>, String>> = std::thread::scope(|s| {
        let hs: Vec<_> = rows
            .chunks(chunk)
            .map(|part| {
                let score = &score;
                s.spawn(move || part.iter().map(score).collect())
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("risk-filter worker panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(rows.len());
    for p in parts {
        out.extend(p?);
    }
    Ok(out)
}

/// The rows kept: those whose score is at least `threshold` (a row with no score is never kept).
pub fn keep(rows: &[Row], scores: &[Option<i64>], threshold: i64) -> Vec<Row> {
    rows.iter()
        .zip(scores)
        .filter(|(_, s)| s.is_some_and(|s| s >= threshold))
        .map(|(r, _)| r.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use equity::MoneyChance;
    use nn::encode::Features;

    fn row(trick: u8, tag: u8) -> Row {
        Row {
            features: Features {
                active: [0; nn::encode::MAX_ACTIVE],
                n_active: 0,
                dense: [0; nn::encode::N_DENSE],
            },
            value: MoneyChance::even(),
            source: tag,
            trick,
            label: 0,
            mask: 1,
            outcomes: vec![MoneyChance::even()],
        }
    }

    fn score(r: &Row) -> Result<Option<i64>, String> {
        Ok((r.trick == 3).then_some(i64::from(r.source) * 10))
    }

    #[test]
    fn scores_are_in_row_order_and_thread_independent() {
        let rows: Vec<Row> = (0..1000u32)
            .map(|i| row(if i % 3 == 0 { 2 } else { 3 }, (i % 200) as u8))
            .collect();
        let one = score_rows(&rows, 1, score).unwrap();
        for t in [2, 7, 64, 5000] {
            assert_eq!(score_rows(&rows, t, score).unwrap(), one, "threads {t}");
        }
        assert_eq!(one[0], None, "a trick-2 row has no score");
        assert_eq!(one[1], Some(10));
    }

    #[test]
    fn keeps_only_rows_at_or_above_the_threshold_in_order() {
        let rows: Vec<Row> = (0..6u8).map(|i| row(3, i)).collect();
        let scores = vec![Some(0), None, Some(30), Some(50), Some(29), Some(50)];
        let kept = keep(&rows, &scores, 30);
        let tags: Vec<u8> = kept.iter().map(|r| r.source).collect();
        assert_eq!(tags, vec![2, 3, 5]);
    }

    #[test]
    fn a_failing_row_is_an_error() {
        let rows = vec![row(3, 1), row(3, 2)];
        let bad = |r: &Row| {
            if r.source == 2 {
                Err("row 2: not an encoding".to_string())
            } else {
                Ok(Some(1))
            }
        };
        assert!(score_rows(&rows, 2, bad).is_err());
    }
}
