use super::*;
use crate::array::PosIter;
use std::fmt;

// Dyalog `]display`-style boxes. Each element renders through its own
// `Display` (precision and `#` forwarded); a multi-line rendering is laid out
// as a block, so nested arrays box themselves. Widths count chars, so wide
// glyphs (CJK, emoji) can misalign.
//
//   ┌→────────┐   → last axis, one ↓ per leading axis, ⊖ for an empty axis;
//   ↓0 1  2  3│   ~ every element fits on one line, ∊ some element is a block.
//   │4 5  6  7│
//   └~────────┘

/// Arrays with more elements than this show `EDGE` items at each end of every
/// long axis, as NumPy does.
const THRESHOLD: usize = 1000;
const EDGE: usize = 3;
const ELLIPSIS: &str = "·";

impl<T: fmt::Display, R: Rank, B: Buffer<T>> fmt::Display for Ndarr<T, R, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (buffer, shape, strides) = (self.buffer.as_slice(), self.shape(), self.strides());
        if shape.is_empty() {
            return fmt::Display::fmt(&buffer[self.offset()], f);
        }

        let summarize = self.len() > THRESHOLD;
        let picks: Vec<Vec<Option<usize>>> = shape.iter().map(|&n| picks(n, summarize)).collect();
        let dims: Vec<usize> = picks.iter().map(Vec::len).collect();
        let total = dims.iter().product();

        // `None` marks an elided position.
        let mut cells = Vec::with_capacity(total);
        let mut walk = PosIter::new(&dims, &vec![0; dims.len()], 0, total);
        for _ in 0..total {
            let mut position = Some(self.offset() as isize);
            for (axis, &i) in walk.coordinates().iter().enumerate() {
                position = position
                    .zip(picks[axis][i])
                    .map(|(p, k)| p + k as isize * strides[axis]);
            }
            cells.push(position.map(|p| render(&buffer[p as usize], f)));
            walk.next();
        }

        let nested = cells.iter().flatten().any(|lines| lines.len() > 1);
        let body = layout(&dims, &picks, &cells, f.width().unwrap_or(0));
        f.write_str(&frame(&body, shape, nested).join("\n"))
    }
}

/// Indices shown along an axis of length `n`; `None` is the ellipsis slot.
fn picks(n: usize, summarize: bool) -> Vec<Option<usize>> {
    if summarize && n > 2 * EDGE {
        let mut out: Vec<Option<usize>> = (0..EDGE).map(Some).collect();
        out.push(None);
        out.extend((n - EDGE..n).map(Some));
        out
    } else {
        (0..n).map(Some).collect()
    }
}

/// One element's lines, forwarding the caller's precision and `#` flag.
fn render<T: fmt::Display>(x: &T, f: &fmt::Formatter<'_>) -> Vec<String> {
    let text = match (f.precision(), f.alternate()) {
        (Some(p), true) => format!("{x:#.p$}"),
        (Some(p), false) => format!("{x:.p$}"),
        (None, true) => format!("{x:#}"),
        (None, false) => x.to_string(),
    };
    text.split('\n').map(str::to_owned).collect()
}

fn width(s: &str) -> usize {
    s.chars().count()
}

fn pad_left(s: &str, w: usize) -> String {
    format!("{}{s}", " ".repeat(w.saturating_sub(width(s))))
}

fn pad_right(s: &str, w: usize) -> String {
    format!("{s}{}", " ".repeat(w.saturating_sub(width(s))))
}

/// Numbers right-align on their decimal point; anything else left-aligns.
fn is_numeric(s: &str) -> bool {
    s.starts_with(|c: char| c.is_ascii_digit() || "+-.".contains(c)) || s == "NaN" || s == "inf"
}

/// How one column renders its cells.
enum Column {
    Numeric { int: usize, frac: usize },
    Text(usize),
}

impl Column {
    fn new<'a>(cells: impl Iterator<Item = &'a Option<Vec<String>>>, min: usize) -> Self {
        let shown: Vec<&Vec<String>> = cells.flatten().collect();
        let numeric = !shown.is_empty()
            && shown
                .iter()
                .all(|lines| lines.len() == 1 && is_numeric(&lines[0]));
        if numeric {
            let (mut int, mut frac) = (0, 0);
            for lines in shown {
                let (i, d) = split_decimal(&lines[0]);
                int = int.max(width(i));
                frac = frac.max(width(d));
            }
            Column::Numeric {
                int: int.max(min.saturating_sub(frac)),
                frac,
            }
        } else {
            let w = shown
                .iter()
                .flat_map(|lines| lines.iter())
                .map(|l| width(l))
                .max();
            Column::Text(w.unwrap_or(0).max(min).max(width(ELLIPSIS)))
        }
    }

    fn width(&self) -> usize {
        match *self {
            Column::Numeric { int, frac } => int + frac,
            Column::Text(w) => w,
        }
    }

    /// Line `k` of a cell, padded to the column width.
    fn line(&self, cell: &Option<Vec<String>>, k: usize) -> String {
        match (self, cell) {
            (Column::Numeric { int, frac }, Some(lines)) if k == 0 => {
                let (i, d) = split_decimal(&lines[0]);
                pad_left(i, *int) + &pad_right(d, *frac)
            }
            (Column::Text(w), Some(lines)) => pad_right(lines.get(k).map_or("", |l| l), *w),
            (_, None) if k == 0 => pad_left(ELLIPSIS, self.width()),
            _ => " ".repeat(self.width()),
        }
    }
}

fn split_decimal(s: &str) -> (&str, &str) {
    s.find('.').map_or((s, ""), |i| s.split_at(i))
}

/// The box interior: the last axis runs across, the one before it down, and
/// each step of a higher axis adds a blank line between 2-D slices.
fn layout(
    dims: &[usize],
    picks: &[Vec<Option<usize>>],
    cells: &[Option<Vec<String>>],
    min: usize,
) -> Vec<String> {
    let rank = dims.len();
    let cols = dims[rank - 1];
    let rows = if rank >= 2 { dims[rank - 2] } else { 1 };
    let lead = &dims[..rank.saturating_sub(2)];
    if cols == 0 && rank == 1 {
        return Vec::new();
    }
    let columns: Vec<Column> = (0..cols)
        .map(|j| Column::new(cells.iter().skip(j).step_by(cols.max(1)), min))
        .collect();

    let mut lines = Vec::new();
    let slices: usize = lead.iter().product();
    let mut walk = PosIter::new(lead, &vec![0; lead.len()], 0, slices);
    let mut previous: Option<Vec<usize>> = None;
    for slice in 0..slices {
        let here = walk.coordinates().to_vec();
        walk.next();
        if let Some(previous) = previous {
            let changed = previous
                .iter()
                .zip(&here)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            lines.extend(std::iter::repeat_n(String::new(), lead.len() - changed));
        }
        previous = Some(here.clone());

        if here
            .iter()
            .enumerate()
            .any(|(axis, &i)| picks[axis][i].is_none())
        {
            lines.push(ELLIPSIS.to_owned());
            continue;
        }
        for r in 0..rows {
            let row = &cells[(slice * rows + r) * cols..][..cols];
            let height = row.iter().flatten().map(Vec::len).max().unwrap_or(1);
            for k in 0..height {
                let parts: Vec<String> = columns
                    .iter()
                    .zip(row)
                    .map(|(c, cell)| c.line(cell, k))
                    .collect();
                lines.push(parts.join(" "));
            }
        }
    }
    lines
}

/// Box `body` with Dyalog's corners and axis markers.
fn frame(body: &[String], shape: &[usize], nested: bool) -> Vec<String> {
    let rank = shape.len();
    let marker = |n: usize, mark: &str| {
        if n == 0 {
            "⊖".to_owned()
        } else {
            mark.to_owned()
        }
    };
    let inner = body.iter().map(|l| width(l)).max().unwrap_or(0).max(1);
    let depth = rank.saturating_sub(1).max(1);
    let rule = "─".repeat(inner - 1);

    let mut out = vec![format!(
        "{}{}{rule}┐",
        "┌".repeat(depth),
        marker(shape[rank - 1], "→")
    )];
    let mut body: Vec<&str> = body.iter().map(String::as_str).collect();
    if body.is_empty() && rank >= 2 {
        body.push(""); // somewhere to show the leading markers
    }
    for (i, line) in body.iter().enumerate() {
        let lead = if i == 0 && rank >= 2 {
            shape[..rank - 1].iter().map(|&n| marker(n, "↓")).collect()
        } else {
            "│".repeat(depth)
        };
        out.push(format!("{lead}{}│", pad_right(line, inner)));
    }
    out.push(format!(
        "{}{}{rule}┘",
        "└".repeat(depth),
        if nested { "∊" } else { "~" }
    ));
    out
}
