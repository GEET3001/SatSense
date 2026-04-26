import { atom } from 'jotai'

export type Prediction = {
  fee_1block: number
  fee_3block: number
  fee_6block: number
  confidence: number
  model_version: string
  predicted_at: string
  sentiment_score?: number
}

export type FeatureRow = {
  id: string
  captured_at: string
  tx_count: number
  median_fee_rate: number
  total_size_mb: number
  fee_cluster: number
  sentiment_score: number
  sentiment_velocity: number
  article_volume: number
  predictions?: Prediction[]
  prediction?: Prediction
}

export const latestPredictionAtom = atom<Prediction | null>(null)

export const featuresHistoryAtom = atom<FeatureRow[]>([])

export const connectionStatusAtom = atom<'loading' | 'live' | 'error'>('loading')

export const sentimentHistoryAtom = atom<number[]>([])
