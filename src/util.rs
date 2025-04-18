#[derive(Default)]
pub struct Warnings(Vec<Box<dyn std::error::Error>>);

impl Warnings {
    pub fn add(&mut self, warn: Box<dyn std::error::Error>) {
        self.0.push(warn);
    }
}

impl std::fmt::Debug for Warnings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            writeln!(f, "0 warnings.")
        } else {
            writeln!(f, "{} warnings:", self.0.len())?;
            for (ndx, warn) in self.0.iter().enumerate() {
                writeln!(f, "  #{:4}: {}", ndx, warn)?;
            }
            Ok(())
        }
    }
}

pub trait ToWarnings {
    type Ok;
    fn or_warn(self, warnings: &mut Warnings) -> Option<Self::Ok>;
}

impl<T> ToWarnings for std::result::Result<T, anyhow::Error> {
    type Ok = T;

    fn or_warn(self, warnings: &mut Warnings) -> Option<T> {
        match self {
            Ok(value) => Some(value),
            Err(err) => {
                warnings.add(err.into());
                None
            }
        }
    }
}
