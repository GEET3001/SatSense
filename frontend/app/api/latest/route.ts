import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
    // Fall back to anon key if service key is missing or still a placeholder
    const serviceKey = process.env.SUPABASE_SERVICE_KEY
    const supabaseKey =
      serviceKey && serviceKey !== 'your_service_role_key_here'
        ? serviceKey
        : process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

    const supabase = createClient(supabaseUrl, supabaseKey)

    const { data: features, error: featuresError } = await supabase
      .from('features')
      .select('*')
      .order('captured_at', { ascending: false })
      .limit(288)

    if (featuresError) {
      console.error('Supabase features query error:', featuresError)
      return NextResponse.json([], { status: 200 })
    }

    // For each feature, fetch its latest prediction
    const featureIds = (features ?? []).map((f: any) => f.id).filter(Boolean)

    let predictionsByFeatureId: Record<string, any> = {}

    if (featureIds.length > 0) {
      const { data: predictions } = await supabase
        .from('predictions')
        .select('*')
        .in('feature_id', featureIds)
        .order('predicted_at', { ascending: false })

      if (predictions) {
        for (const p of predictions) {
          // Keep only the most recent prediction per feature
          if (!predictionsByFeatureId[p.feature_id]) {
            predictionsByFeatureId[p.feature_id] = p
          }
        }
      }
    }

    const formattedData = (features ?? []).map((f: any) => ({
      ...f,
      prediction: predictionsByFeatureId[f.id] ?? null,
    }))

    return NextResponse.json(formattedData)
  } catch (error: any) {
    console.error('Error in /api/latest:', error)
    // Return empty array so the dashboard renders with skeletons rather than crashing
    return NextResponse.json([], { status: 200 })
  }
}
