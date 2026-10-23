export type AiState =
  | "not_analysed"
  | "model_not_ready"
  | "likely_normal"
  | "likely_anomaly"
  | "ambiguous";

export type HumanDecision =
  | "approved_visual"
  | "rejected_visual"
  | "escalated_ndt"
  | "recapture_requested";

export type BoundingBox = {
  id?: number;
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  confidence?: number | null;
  source: "human" | "model";
};

export type QualityMetrics = {
  reliability?: number;
  blur_laplacian_variance?: number;
  glare_fraction?: number;
  dark_fraction?: number;
  contrast_std?: number;
  reconstruction_error_mean?: number;
  reconstruction_error_p95?: number;
  reconstruction_error_top?: number;
  normal_reference_top_p95?: number;
  normal_reference_top_p99?: number;
  note?: string;
  xai_method?: string;
  xai_interpretation?: string;
  decision_basis?: string[];
  defect_type_status?: string;
  defect_type_model_version?: string;
  predicted_defect_type?: string;
  predicted_defect_type_confidence?: number;
  defect_type_distribution?: Record<string, number>;
  defect_type_predictions?: Array<{
    box_index?: number;
    label: string;
    top_class: string;
    confidence: number;
    distribution: Record<string, number>;
    roi: { x: number; y: number; width: number; height: number };
    source?: string;
  }>;
  defect_type_scan?: {
    candidate_count: number;
    normal_tile_fraction: number;
    strongest_by_type: Record<string, number>;
    conflict: boolean;
  };
  active_model_family?: string;
  detector_status?: string;
  detector_model_version?: string;
  detector_label?: string;
  detector_confidence?: number;
  detector_threshold?: number;
  detector_box_count?: number;
  detector_raw_proposal_count?: number;
  detector_inference_seconds?: number;
  detector_benchmark?: {
    image_level_f1: number;
    image_level_precision: number;
    image_level_recall: number;
    map_50: number;
    map_50_95: number;
    evaluation_scope: string;
  };
  residual_support_status?: string;
  residual_model_version?: string;
  reconstruction_anomaly_index?: number;
};

export type Inspection = {
  id: number;
  created_at: string;
  updated_at: string;
  source_name: string;
  source_path: string;
  source_label?: string | null;
  part_profile: string;
  acquisition_context: Record<string, string>;
  ai_state: AiState;
  anomaly_score?: number | null;
  operational_confidence?: number | null;
  quality_state: "not_assessed" | "acceptable" | "capture_quality_warning";
  quality: QualityMetrics;
  model_version?: string | null;
  model_boxes: BoundingBox[];
  human_boxes: BoundingBox[];
  xai_url?: string | null;
  image_url: string;
  final_decision?: HumanDecision | null;
  final_reason?: string | null;
  final_actor?: string | null;
  final_at?: string | null;
};

export type Dashboard = {
  counts: Record<string, number>;
  latest: Inspection[];
  training: TrainingQueue;
};

export type TrainingVersion = {
  version_name: string;
  model_type: string;
  status: string;
  created_at: string;
  metrics: Record<string, number>;
  notes?: string;
};

export type TrainingQueue = {
  threshold: number;
  primary_normal_learning?: TrainingLoop;
  secondary_rejected_learning?: TrainingLoop;
  reviewed_complex_cases: number;
  processed_complex_cases: number;
  pending_complex_cases: number;
  remaining_to_trigger: number;
  eligible_now: boolean;
  model_versions: TrainingVersion[];
};

export type TrainingLoop = {
  threshold: number;
  reviewed_cases: number;
  processed_cases: number;
  pending_cases: number;
  remaining_to_trigger: number;
  eligible_now: boolean;
  training_target: string;
  eligibility_rule: string;
};

export type AuditEvent = {
  id: number;
  created_at: string;
  actor: string;
  event_type: string;
  payload: Record<string, unknown>;
  previous_hash?: string | null;
  event_hash: string;
};

export type AnnotationRevision = {
  id: number;
  revision_number: number;
  created_at: string;
  actor: string;
  note?: string | null;
  boxes: BoundingBox[];
};
