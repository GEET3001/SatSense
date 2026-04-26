'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'

const terms = [
  {
    title: 'Mempool',
    description: 'The "waiting room" for Bitcoin transactions. Before a transaction is confirmed, it sits in the mempool of every Bitcoin node. Higher congestion means longer wait times.'
  },
  {
    title: 'sat/vB (Sats per Virtual Byte)',
    description: 'The standard unit of Bitcoin transaction fees. Think of it as "price per unit of space" in a block. High-priority transactions pay more sat/vB to skip the line.'
  },
  {
    title: 'Sentiment Score',
    description: 'A numeric value from -1.0 (Extreme Fear/Negative) to +1.0 (Extreme Greed/Positive). It is calculated by analyzing real-time news catalysts and market trends.'
  },
  {
    title: 'Confirmation Targets',
    description: 'Estimates of how many blocks (next, 3, or 6) it will take for your transaction to be mined based on current and predicted congestion.'
  },
  {
    title: 'Fee Clusters',
    description: 'Categories of fee regimes. "Economy" represents retail usage, while "Priority" often signals institutional batches or Layer 2 settlement activity.'
  },
  {
    title: 'Sentiment Velocity',
    description: 'The rate of change in market sentiment. A high velocity means news is breaking fast and market emotion is shifting rapidly, which often precedes fee spikes.'
  }
]

const inferences = [
  {
    signal: 'Bullish Sentiment + Low Congestion',
    inference: 'Ideal for opening L2 channels or consolidating small UTXOs. Retail demand is low, but market confidence is building.',
    color: 'text-green-400'
  },
  {
    signal: 'Bearish Sentiment + Rising Fees',
    inference: 'Typical of a "panic" phase where users are rushing to move funds to exchanges. Expect high volatility and potential fee spikes.',
    color: 'text-red-400'
  },
  {
    signal: 'Neutral Sentiment + Urgent Cluster',
    inference: 'Likely heavy institutional or Layer 2 batching activity. The "base" market is calm, but the mempool is being filled by heavy automated settlement.',
    color: 'text-blue-400'
  }
]

const steps = [
  {
    step: '1. Data Ingestion',
    detail: 'Every 5-10 minutes, the engine scrapes the Bitcoin Mempool for congestion data and over 40 global news sources for market catalysts.'
  },
  {
    step: '2. Sentiment Analysis',
    detail: 'An automated algorithm reads every news headline and assigns a score based on how it will impact Bitcoin fee markets.'
  },
  {
    step: '3. Pattern Recognition',
    detail: 'The system compares current metrics to thousands of historical fee cycles (2020-2026) to find the closest matching market regime.'
  },
  {
    step: '4. Probabilistic Forecasting',
    detail: 'Finally, the system generates three distinct sat/vB targets for "Next Block", "3 Blocks", and "6 Blocks" with a confidence rating.'
  }
]

export default function GuidePage() {
  return (
    <div className="min-h-screen bg-[#0f1117] text-white selection:bg-gray-800">
      <header className="flex h-12 w-full flex-row items-center justify-between border-b border-gray-800 px-6">
        <Link href="/" className="text-lg font-medium text-white hover:text-blue-400 transition-colors">
          ← Back to Dashboard
        </Link>
        <div className="text-sm text-gray-500 uppercase tracking-widest font-semibold">User Guide</div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-12">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            Understanding the Dashboard
          </h1>
          <p className="text-gray-400 text-lg">
            Welcome to the 2026 Bitcoin economy. This guide explains how to read our real-time metrics and translate them into actionable market insights.
          </p>
        </motion.div>

        {/* TERMINOLOGY SECTION */}
        <section className="mb-16">
          <h2 className="text-2xl font-semibold mb-8 border-l-4 border-blue-500 pl-4">Core Terminology</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {terms.map((term, idx) => (
              <motion.div 
                key={term.title}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="p-6 rounded-xl border border-gray-800 bg-[#161b27] hover:border-gray-600 transition-all"
              >

                <h3 className="text-lg font-bold text-white mb-2">{term.title}</h3>
                <p className="text-sm text-gray-400 leading-relaxed">{term.description}</p>
              </motion.div>
            ))}
          </div>
        </section>

        {/* INFERENCE SECTION */}
        <section className="mb-16">
          <h2 className="text-2xl font-semibold mb-8 border-l-4 border-purple-500 pl-4">Market Inferences</h2>
          <div className="space-y-4">
            {inferences.map((inf, idx) => (
              <motion.div 
                key={inf.signal}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5 + idx * 0.1 }}
                className="p-6 rounded-xl border border-gray-800 bg-[#161b27]"
              >
                <h3 className={`text-sm font-mono uppercase tracking-wider mb-2 ${inf.color}`}>
                  Signal: {inf.signal}
                </h3>
                <p className="text-white text-lg">
                  {inf.inference}
                </p>
              </motion.div>
            ))}
          </div>
        </section>

        {/* PREDICTION PROCESS SECTION */}
        <section className="mb-16">
          <h2 className="text-2xl font-semibold mb-8 border-l-4 border-amber-500 pl-4">The Prediction Engine</h2>
          <div className="relative border-l border-gray-800 ml-4 pl-8 space-y-12">
            {steps.map((item, idx) => (
              <motion.div 
                key={item.step}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.8 + idx * 0.1 }}
                className="relative"
              >
                <div className="absolute -left-[41px] top-0 w-4 h-4 rounded-full bg-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.5)]"></div>
                <h3 className="text-amber-400 font-mono text-sm mb-1 uppercase tracking-tighter">{item.step}</h3>
                <p className="text-white text-base leading-relaxed">{item.detail}</p>
              </motion.div>
            ))}
          </div>
        </section>


      </main>

      <footer className="py-12 border-t border-gray-800 text-center text-gray-600 text-sm">
        &copy; 2026 SatSense Analytics. For informational purposes only.
      </footer>
    </div>
  )
}
