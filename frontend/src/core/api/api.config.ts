// ===================
// © AngelaMos | 2026
// api.config.ts
// ===================

import axios, { type AxiosError, type AxiosInstance } from 'axios'
import { transformAxiosError } from './errors'

const getBaseURL = (): string => {
  return import.meta.env.VITE_API_URL ?? '/api'
}

export const apiClient: AxiosInstance = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 15000,
  headers: { "X-API-Key": "dev-key" },
})

apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError): Promise<never> => {
    return Promise.reject(transformAxiosError(error))
  }
)
