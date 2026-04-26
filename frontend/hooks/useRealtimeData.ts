'use client'

import { useEffect } from 'react'
import { useSetAtom } from 'jotai'
import supabase from '../lib/supabaseClient'
import {
  latestPredictionAtom,
  featuresHistoryAtom,
  connectionStatusAtom,
  sentimentHistoryAtom,
  FeatureRow,
  Prediction
} from '../lib/atoms'

export function useRealtimeData() {
  const setLatestPrediction = useSetAtom(latestPredictionAtom)
  const setFeaturesHistory = useSetAtom(featuresHistoryAtom)
  const setConnectionStatus = useSetAtom(connectionStatusAtom)
  const setSentimentHistory = useSetAtom(sentimentHistoryAtom)

  useEffect(() => {
    let mounted = true

    async function fetchLatest() {
      try {
        const res = await fetch('/api/latest')
        if (!res.ok) throw new Error('Failed to fetch latest data')
        const data = await res.json()

        if (mounted) {
          setFeaturesHistory(data)

          const sentiments = data
            .slice(0, 24)
            .map((d: FeatureRow) => d.sentiment_score)
            .filter((s: number) => s !== undefined && s !== null)
          setSentimentHistory(sentiments)

          if (data.length > 0 && data[0].predictions && data[0].predictions.length > 0) {
            setLatestPrediction(data[0].predictions[0])
          } else if (data.length > 0 && data[0].prediction) {
            setLatestPrediction(data[0].prediction)
          }

          setConnectionStatus('live')
        }
      } catch (err) {
        console.error(err)
        if (mounted) setConnectionStatus('error')
      }
    }

    fetchLatest()

    const channel = supabase
      .channel('features-channel')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'features' },
        (payload) => {
          const newRow = payload.new as FeatureRow

          setFeaturesHistory((prev) => {
            const next = [newRow, ...prev]
            return next.slice(0, 288)
          })

          if (newRow.sentiment_score !== undefined) {
            setSentimentHistory((prev) => {
              const next = [newRow.sentiment_score, ...prev]
              return next.slice(0, 24)
            })
          }

          if (newRow.prediction) {
            setLatestPrediction(newRow.prediction)
          } else if (newRow.predictions && newRow.predictions.length > 0) {
            setLatestPrediction(newRow.predictions[0])
          }
        },
      )
      .subscribe((status) => {
        if (status === 'SUBSCRIBED') setConnectionStatus('live')
        if (status === 'CHANNEL_ERROR' || status === 'TIMED_OUT') setConnectionStatus('error')
      })

    return () => {
      mounted = false
      supabase.removeChannel(channel)
    }
  }, [setLatestPrediction, setFeaturesHistory, setConnectionStatus, setSentimentHistory])
}
