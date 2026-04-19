use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ForestFile {
    n_features: usize,
    n_classes: usize,
    classes: Vec<i32>,
    trees: Vec<TreeFile>,
}

#[derive(Debug, Deserialize)]
struct TreeFile {
    children_left: Vec<i32>,
    children_right: Vec<i32>,
    feature: Vec<i32>,
    threshold: Vec<f64>,
    values: Vec<Vec<f64>>,
}

#[derive(Debug)]
struct TreeModel {
    children_left: Vec<i32>,
    children_right: Vec<i32>,
    feature: Vec<i32>,
    threshold: Vec<f64>,
    values: Vec<Vec<f64>>,
}

#[derive(Debug)]
pub struct RandomForestRunner {
    n_features: usize,
    n_classes: usize,
    classes: Vec<i32>,
    trees: Vec<TreeModel>,
}

impl TreeModel {
    fn predict_proba(&self, row: &[f32], n_classes: usize) -> Option<Vec<f32>> {
        let mut node: usize = 0;
        loop {
            let left = *self.children_left.get(node)?;
            let right = *self.children_right.get(node)?;

            if left == -1 && right == -1 {
                let counts = self.values.get(node)?;
                if counts.len() != n_classes {
                    return None;
                }

                let total: f64 = counts.iter().sum();
                if total <= 0.0 {
                    return Some(vec![0.0; n_classes]);
                }

                let mut proba = vec![0.0_f32; n_classes];
                for (i, &count) in counts.iter().enumerate() {
                    proba[i] = (count / total) as f32;
                }
                return Some(proba);
            }

            let feature_idx = *self.feature.get(node)?;
            if feature_idx < 0 {
                return None;
            }
            let feature_idx = feature_idx as usize;
            let threshold = *self.threshold.get(node)?;

            let feature_value = *row.get(feature_idx)? as f64;

            node = if feature_value <= threshold {
                left as usize
            } else {
                right as usize
            };
        }
    }
}

impl RandomForestRunner {
    pub fn from_json(model_json: &str) -> Option<Self> {
        let file: ForestFile = serde_json::from_str(model_json).ok()?;

        let mut trees = Vec::with_capacity(file.trees.len());
        for t in file.trees {
            let node_count = t.children_left.len();
            if t.children_right.len() != node_count
                || t.feature.len() != node_count
                || t.threshold.len() != node_count
                || t.values.len() != node_count
            {
                return None;
            }
            trees.push(TreeModel {
                children_left: t.children_left,
                children_right: t.children_right,
                feature: t.feature,
                threshold: t.threshold,
                values: t.values,
            });
        }

        Some(Self {
            n_features: file.n_features,
            n_classes: file.n_classes,
            classes: file.classes,
            trees,
        })
    }

    pub fn predict_probabilities(&self, feature_rows: &[Vec<f32>]) -> Option<Vec<Vec<f32>>> {
        if feature_rows.is_empty() {
            return Some(Vec::new());
        }

        let mut all_probs = Vec::with_capacity(feature_rows.len());

        for row in feature_rows {
            if row.len() != self.n_features {
                return None;
            }

            let mut sum_probs = vec![0.0_f32; self.n_classes];
            for tree in &self.trees {
                let tree_probs = tree.predict_proba(row, self.n_classes)?;
                for (i, p) in tree_probs.iter().enumerate() {
                    sum_probs[i] += *p;
                }
            }

            let inv = 1.0_f32 / self.trees.len() as f32;
            for p in &mut sum_probs {
                *p *= inv;
            }
            all_probs.push(sum_probs);
        }

        Some(all_probs)
    }

    pub fn class_ids(&self) -> &[i32] {
        &self.classes
    }
}
