'use client'

import { useEffect, useState } from 'react'

type NewsItem = {
  source: string
  text: string
  score: number
}

export default function NewsFeed() {
  const [news, setNews] = useState<NewsItem[]>([])
  const [loading, setLoading] = useState(true)

  const fetchNews = async () => {
    try {
      const res = await fetch('/api/news')
      if (res.ok) {
        const data = await res.json()
        setNews(data)
      }
    } catch (err) {
      console.error('Failed to fetch news', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchNews()
    const interval = setInterval(fetchNews, 60000) // refresh every minute
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="rounded-xl border border-gray-800 bg-[#161b27] p-6 animate-pulse">
        <div className="h-6 w-48 bg-gray-700 rounded mb-4"></div>
        <div className="space-y-3">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="h-16 bg-gray-800 rounded"></div>
          ))}
        </div>
      </div>
    )
  }

  if (news.length === 0) {
    return null
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-[#161b27] p-6 mt-6">
      <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-400"><path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"></path><path d="M18 14h-8"></path><path d="M15 18h-5"></path><path d="M10 6h8v4h-8V6Z"></path></svg>
        Latest News
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {news.map((item, idx) => {
          const isPos = item.score > 0.2
          const isNeg = item.score < -0.2
          const badgeColor = isPos ? 'bg-green-500/10 text-green-400 border-green-500/20' : 
                             isNeg ? 'bg-red-500/10 text-red-400 border-red-500/20' : 
                             'bg-gray-500/10 text-gray-400 border-gray-500/20'
          
          return (
            <div key={idx} className="group flex flex-col justify-between rounded-lg border border-gray-800 bg-[#1a2130] p-4 transition-all hover:border-gray-600 hover:bg-[#1e2638]">
              <p className="text-sm text-gray-300 mb-3 line-clamp-3">{item.text}</p>
              <div className="flex items-center justify-between mt-auto">
                <span className="text-xs text-gray-500 font-mono uppercase tracking-wider">{item.source}</span>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400">Sentiment Impact:</span>
                  <span className={`text-xs px-2 py-1 rounded border ${badgeColor} font-mono`}>
                    {item.score > 0 ? '+' : ''}{item.score.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
