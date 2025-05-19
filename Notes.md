## 2025/5/17 Motion model matched to target

If the motion model is well matched to target dynamics and NIS is
consistent with the Chi-squared distribution, thresholds for gating
and data association (which use Mahalanobis distances) can be derived
from the Chi-squared distribution and represent a specific probability
of rejecting a correct association.

For example, if the threshols is set as the 0.95 quantile of the
Chi-squared distribution with 3 degrees of freedom, then about 5% of
correct measurements will be rejected.


## 2025/5/3 Motion models vs. target models

* NCV(1) target shows perfect NEES match with the CV(1) motion model;
  ~95% of mean NEES falls into the 95% confidence interval
* for the CV target and CV(1) motion model (truly constant-velocity
  target, no noise), there is an exchange between NEES being within
  the confidence interval (which means that the reported covariance
  matrix accurately represents the actual estimation error) and the
  actual error; this happens for mismatched motion / target models
