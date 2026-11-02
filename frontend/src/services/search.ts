import type {
  HealthResponse,
  SearchRequest,
  SearchResponse,
  VideoUploadResponse,
} from '../types/search'

const configuredBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim().replace(/\/+$/, '')

export class ApiError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message)
    this.name = 'ApiError'
  }
}

function apiUrl(path: string) {
  return `${configuredBaseUrl ?? ''}${path}`
}

async function readResponse<T>(response: Response): Promise<T> {
  if (response.ok) return response.json() as Promise<T>

  let message = `请求失败（HTTP ${response.status}）`
  try {
    const body = (await response.json()) as { detail?: unknown }
    if (typeof body.detail === 'string') message = body.detail
    else if (Array.isArray(body.detail)) {
      const details = body.detail.map((item) => {
        if (typeof item?.msg !== 'string') return ''
        const field = Array.isArray(item.loc) ? item.loc.slice(1).join('.') : ''
        return field ? `${field}: ${item.msg}` : item.msg
      }).filter(Boolean)
      if (details.length) message = details.join('；')
    }
  } catch {
    // Keep the HTTP status message when the server did not return JSON.
  }
  throw new ApiError(message, response.status)
}

export async function uploadVideo(file: File, duration: number): Promise<VideoUploadResponse> {
  const body = new FormData()
  body.append('file', file)
  body.append('duration', String(duration))

  const response = await fetch(apiUrl('/api/videos'), {
    method: 'POST',
    body,
  })
  return readResponse<VideoUploadResponse>(response)
}

export async function searchVideo(
  request: SearchRequest,
): Promise<SearchResponse> {
  const response = await fetch(apiUrl('/api/search'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  return readResponse<SearchResponse>(response)
}

export async function getBackendHealth(): Promise<HealthResponse> {
  const response = await fetch(apiUrl('/api/health'))
  return readResponse<HealthResponse>(response)
}
