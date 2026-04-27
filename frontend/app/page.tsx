'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useAtomValue } from 'jotai'
import { useRealtimeData } from '@/hooks/useRealtimeData'
import {
  latestPredictionAtom,
  featuresHistoryAtom,
  connectionStatusAtom,
  sentimentHistoryAtom,
  Prediction,
  FeatureRow
} from '@/lib/atoms'
import MempoolChart from '@/components/charts/MempoolChart'
import SentimentSparkline from '@/components/charts/SentimentSparkline'
import NewsFeed from '@/components/NewsFeed'

export default function Dashboard() {
  useRealtimeData()

  const latestPrediction = useAtomValue(latestPredictionAtom)
  const featuresHistory = useAtomValue(featuresHistoryAtom)
  const connectionStatus = useAtomValue(connectionStatusAtom)
  const sentimentHistory = useAtomValue(sentimentHistoryAtom)

  const [clock, setClock] = useState('')

  useEffect(() => {
    setClock(new Date().toUTCString())
    const interval = setInterval(() => {
      setClock(new Date().toUTCString())
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  const feeCards = [
    { label: 'Next block', key: 'fee_1block', time: '≈10 min' },
    { label: '~3 blocks', key: 'fee_3block', time: '≈30 min' },
    { label: '~6 blocks', key: 'fee_6block', time: '≈60 min' },
  ]

  const latest = featuresHistory[0] as FeatureRow | undefined
  const latestPred = latestPrediction as Prediction | null
  const sentimentScore = latestPred?.sentiment_score ?? latest?.sentiment_score ?? 0
  const isBullish = sentimentScore > 0.2
  const isBearish = sentimentScore < -0.2
  const sentimentColor = isBullish ? 'text-green-400' : isBearish ? 'text-red-400' : 'text-gray-400'
  const sentimentLabel = isBullish ? 'Bullish' : isBearish ? 'Bearish' : 'Neutral'


  const clusterNames: Record<number, string> = {
    0: 'Low priority',
    1: 'Economy',
    2: 'Normal',
    3: 'Priority',
    4: 'Urgent',
  }
  const clusterColors: Record<number, string> = {
    0: 'text-gray-500',
    1: 'text-blue-500',
    2: 'text-green-500',
    3: 'text-amber-500',
    4: 'text-red-500',
  }
  const clusterKey = latest?.fee_cluster ?? 2

  return (
    <div className="min-h-screen bg-[#0f1117] text-white selection:bg-gray-800">
      {/* HEADER */}
      <header className="flex h-12 w-full flex-row items-center justify-between border-b border-gray-800 px-6">
        <div>
          <span className="text-lg font-bold text-white tracking-tight">SatSense</span>
        </div>
        <div>
          <span className="font-mono text-sm text-gray-400">{clock}</span>
          <Link href="/guide" className="ml-6 text-xs font-semibold text-blue-400 hover:text-blue-300 transition-colors uppercase tracking-widest border border-blue-500/30 px-3 py-1 rounded-full bg-blue-500/5">
            View Guide
          </Link>
        </div>
        <div className="flex items-center gap-2">
          {connectionStatus === 'live' && (
            <>
              <span className="h-2 w-2 animate-pulse rounded-full bg-green-500"></span>
              <span className="text-sm text-green-500">Live</span>
            </>
          )}
          {connectionStatus === 'loading' && (
            <>
              <span className="h-2 w-2 animate-pulse rounded-full bg-gray-500"></span>
              <span className="text-sm text-gray-400">Connecting</span>
            </>
          )}
          {connectionStatus === 'error' && (
            <>
              <span className="h-2 w-2 rounded-full bg-red-500"></span>
              <span className="text-sm text-red-500">Error</span>
            </>
          )}
        </div>
      </header>

      {/* FEE CARDS */}
      <div className="grid grid-cols-1 gap-4 px-6 py-4 md:grid-cols-3">
        {feeCards.map((card) => {
          const val = latestPred ? (latestPred as any)[card.key] : null
          const confidence = latestPred?.confidence ?? 0
          const confPercent = confidence * 100

          let barColor = 'bg-red-500'
          if (confidence > 0.7) barColor = 'bg-green-500'
          else if (confidence > 0.4) barColor = 'bg-amber-500'

          return (
            <div key={card.key} className="rounded-xl border border-gray-800 bg-[#161b27] p-5">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-sm text-gray-400">{card.label}</span>
                <span className="text-xs text-gray-600">{card.time}</span>
              </div>

              {!latestPrediction ? (
                <div className="h-10 w-24 animate-pulse rounded bg-gray-800"></div>
              ) : (
                <div className="mb-1 flex items-baseline">
                  <span className="font-mono text-4xl font-bold text-white">
                    {val?.toFixed(1) ?? '—'}
                  </span>
                  <span className="ml-1 text-sm text-gray-500">sat/vB</span>
                </div>
              )}

              {latestPrediction && (
                <div className="mt-3 h-1 w-full overflow-hidden rounded bg-gray-800">
                  <div
                    className={`h-full ${barColor} transition-all duration-500`}
                    style={{ width: `${Math.max(confPercent, 0)}%` }}
                  />
                </div>
              )}

              {latestPrediction?.model_version === 'fallback-rules' && (
                <div className="mt-3">
                  <span className="inline-block rounded-full bg-yellow-900/40 px-2 py-0.5 text-xs text-yellow-400">
                    Rule-based estimate
                  </span>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* MAIN CONTENT */}
      <div className="grid grid-cols-1 gap-4 px-6 pb-4 lg:grid-cols-[60%_40%]">
        {/* Left card */}
        <div className="rounded-xl border border-gray-800 bg-[#161b27] p-4">
          <h2 className="mb-3 text-sm text-gray-400">Mempool congestion</h2>
          <MempoolChart data={featuresHistory || []} />
        </div>

        {/* Right card */}
        <div className="rounded-xl border border-gray-800 bg-[#161b27] p-4">
          <h2 className="mb-3 text-sm text-gray-400">Market sentiment</h2>

          <div className="flex flex-col">
            <span className={`${sentimentColor} font-mono text-5xl font-bold`}>
              {sentimentScore.toFixed(3)}
            </span>
            <span
              className={`${sentimentColor} mt-1 text-sm font-semibold uppercase tracking-widest`}
            >
              {sentimentLabel}
            </span>
          </div>

          <div className="my-3 border-t border-gray-800"></div>

          <div className="mb-2 text-xs text-gray-600">24h trend</div>
          <SentimentSparkline data={sentimentHistory || []} score={sentimentScore} />
        </div>
      </div>

      {/* STATS ROW */}
      <div className="grid grid-cols-1 gap-4 px-6 pb-6 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl border border-gray-800 bg-[#161b27] p-4">
          <div className="mb-1 text-xs text-gray-500">Pending txs</div>
          <div className="font-mono text-xl font-medium text-white">
            {latest?.tx_count?.toLocaleString() ?? '—'}
          </div>
        </div>

        <div className="rounded-xl border border-gray-800 bg-[#161b27] p-4">
          <div className="mb-1 text-xs text-gray-500">Mempool size</div>
          <div className="font-mono text-xl font-medium text-white">
            {latest?.total_size_mb !== undefined ? `${latest.total_size_mb.toFixed(1)} MB` : '—'}
          </div>
        </div>

        <div className="rounded-xl border border-gray-800 bg-[#161b27] p-4">
          <div className="mb-1 text-xs text-gray-500">News volume</div>
          <div className="font-mono text-xl font-medium text-white">
            {latest?.article_volume !== undefined ? `${latest.article_volume} articles` : '—'}
          </div>
        </div>

        <div className="rounded-xl border border-gray-800 bg-[#161b27] p-4">
          <div className="mb-1 text-xs text-gray-500">Fee cluster</div>
          <div
            className={`font-mono text-xl font-medium ${clusterColors[clusterKey] || 'text-white'}`}
          >
            {clusterNames[clusterKey] ?? '—'}
          </div>
        </div>
      </div>

      {/* LATEST NEWS / CATALYSTS */}
      <div className="px-6 pb-6">
        <NewsFeed />
      </div>
    </div>
  )
}
