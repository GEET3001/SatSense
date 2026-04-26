'use client'

import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from 'recharts'

export default function MempoolChart({ data }: { data: any[] }) {
  const chartData = [...data].reverse()

  return (
    <div style={{ width: '100%', height: 280 }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" />
          <XAxis
            dataKey="captured_at"
            tickFormatter={(v) =>
              new Date(v).toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit' })
            }
            interval="preserveStartEnd"
            tick={{ fill: '#6B7280', fontSize: 11 }}
          />
          <YAxis
            yAxisId="left"
            tick={{ fill: '#6B7280', fontSize: 11 }}
            label={{ value: 'Transactions', angle: -90, position: 'insideLeft', fill: '#6B7280', fontSize: 12, offset: 10 }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={{ fill: '#F59E0B', fontSize: 11 }}
            label={{ value: 'sat/vB', angle: 90, position: 'insideRight', fill: '#F59E0B', fontSize: 12, offset: 10 }}
          />
          <YAxis
            yAxisId="size"
            orientation="right"
            tick={{ fill: '#3B82F6', fontSize: 11 }}
            label={{ value: 'MB', angle: 90, position: 'insideRight', fill: '#3B82F6', fontSize: 12, offset: 40 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1F2937',
              border: '1px solid #374151',
              color: 'white',
              borderRadius: '8px',
              fontSize: '12px',
            }}
            labelFormatter={(v) => `Time: ${new Date(v).toLocaleString()}`}
            itemStyle={{ padding: '2px 0' }}
          />
          <Bar yAxisId="left" dataKey="tx_count" fill="#374151" name="Transactions" />
          <Line
            yAxisId="right"
            dataKey="median_fee_rate"
            stroke="#F59E0B"
            dot={false}
            strokeWidth={2}
            name="Fee (sat/vB)"
          />
          <Line
            yAxisId="size"
            dataKey="total_size_mb"
            stroke="#3B82F6"
            dot={false}
            strokeWidth={2}
            name="Size (MB)"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
