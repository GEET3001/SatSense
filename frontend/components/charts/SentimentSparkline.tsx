'use client'

import { LineChart, Line, ResponsiveContainer } from 'recharts'

export default function SentimentSparkline({ data, score }: { data: number[]; score: number }) {
  let color = '#6B7280'
  if (score > 0.2) color = '#10B981'
  else if (score < -0.2) color = '#EF4444'

  const chartData = data.map((s, i) => ({ value: s, i })).reverse()

  return (
    <div style={{ width: '100%', height: 60 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <Line
            dataKey="value"
            stroke={color}
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
