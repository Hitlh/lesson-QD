export interface Moment {
  start: number
  end: number
  score: number
  raw_score?: number
}

export interface SearchRequest {
  video_id: string
  query: string
  top_k: number
  nms_method?: 'none' | 'hard' | 'soft_linear'
  nms_threshold?: number
  nms_min_score?: number
}

export interface VideoUploadResponse {
  video_id: string
  filename: string
  duration: number
  status: 'processing' | 'ready' | 'failed'
  stream_url: string
  inference_mode: string
}

export interface HealthResponse {
  status: 'ready' | 'degraded'
  model_loaded: boolean
  device: string
  checkpoint: string
  feature_mode: 'clip' | 'slowfast_clip'
  model_label: string
  slowfast_ready: boolean
  max_video_duration?: number
  max_upload_mb?: number
  postprocessing?: {
    method: 'none' | 'hard' | 'soft_linear'
    iou_threshold: number
    min_score: number
  }
  windowing?: {
    window_seconds: number
    overlap_seconds: number
    feature_chunk_seconds: number
  }
  window_ranking?: { configured: boolean; loaded: boolean; ready: boolean }
}

export interface SearchResponse {
  video_id: string
  query: string
  duration: number
  moments: Moment[]
  saliency_scores: number[]
  clip_duration: number
  model_version: string
  video_feature_cache_hit: boolean
  inference_time_ms: number
  postprocessing?: {
    method: 'none' | 'hard' | 'soft_linear'
    iou_threshold: number
    min_score: number
    candidates_before: number
    candidates_valid: number
    candidates_after: number
    returned_count: number
  }
  windowing?: {
    enabled: boolean
    window_count: number
    window_seconds: number
    overlap_seconds: number
    feature_clip_count: number
    score_aggregation: 'raw_model_score' | 'window_reranked'
    saliency_aggregation: 'mean'
    windows?: {
      window_id: number
      start: number
      end: number
      candidate_count: number
      relevance_logit?: number
    }[]
  }
  window_ranking?: { enabled: boolean; method: 'none' | 'window_head'; strength?: number }
}
