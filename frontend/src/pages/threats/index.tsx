// ===================
// ThreatShield AI | 2026
// index.tsx
// ===================

import { useState } from 'react'
import { useAlerts } from '@/api/hooks'
import type { ThreatEvent, WebSocketAlert } from '@/api/types'
import { MethodBadge, SeverityBadge, ThreatDetail } from '@/components'
import { PAGINATION } from '@/config'
import styles from './threats.module.scss'

type SeverityFilter = 'ALL' | 'HIGH' | 'MEDIUM' | 'LOW'

function formatTime(dateStr: string): string {
  return new Date(dateStr).toLocaleString()
}

export function Component(): React.ReactElement {
  const [offset, setOffset] = useState(0)
  const [severity, setSeverity] = useState<SeverityFilter>('ALL')
  const [sourceIp, setSourceIp] = useState('')
  const [selectedThreat, setSelectedThreat] = useState<ThreatEvent | null>(null)

  const { alerts } = useAlerts()
  const filteredItems = alerts.filter((alert) => {
    const severityMatch =
      severity === 'ALL' ||
      alert.severity.toUpperCase() === severity

    const ipMatch =
      sourceIp === '' ||
      alert.source_ip.includes(sourceIp)
    return severityMatch && ipMatch
  })
  const selectedIndex = selectedThreat
    ? filteredItems.findIndex((item) => item.id === selectedThreat.id)
    : -1
  const handleNext = () => {
    if (selectedIndex < filteredItems.length - 1) {
      setSelectedThreat(alertToThreatEvent(filteredItems[selectedIndex + 1]))
    }
  }
  const handlePrevious = () => {
    if (selectedIndex > 0) {
      setSelectedThreat(alertToThreatEvent(filteredItems[selectedIndex - 1]))
    }
  }


const total = filteredItems.length

const items = filteredItems.slice(
  offset,
  offset + PAGINATION.DEFAULT_LIMIT
)
const showing = Math.min(
  offset + PAGINATION.DEFAULT_LIMIT,
  total
)
const hasPrev = offset > 0

const hasNext =
  offset + PAGINATION.DEFAULT_LIMIT < total

const isLoading = false

  return (
    <div className={styles.page}>
      <div className={styles.filters}>
        <select
          className={styles.select}
          value={severity}
          onChange={(e) => {
            setSeverity(e.target.value as SeverityFilter)
            setOffset(0)
          }}>
          <option value="ALL">All Severities</option>
          <option value="HIGH">High</option>
          <option value="MEDIUM">Medium</option>
          <option value="LOW">Low</option>
        </select>

        <input
          className={styles.input}
          type="text"
          placeholder="Filter by source IP..."
          value={sourceIp}
          onChange={(e) => {
            setSourceIp(e.target.value)
            setOffset(0)
          }
        }
        />
      </div>

      <div className={styles.tableWrapper}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Time</th>
              <th>Source IP</th>
              <th>Method</th>
              <th>Path</th>
              <th>Score</th>
              <th>Severity</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={7} className={styles.loadingCell}>
                  Loading threats...
                </td>
              </tr>
            ) : items.length === 0 ? (
              <tr>
                <td colSpan={7} className={styles.emptyCell}>
                  No threats found matching your filters
                </td>
              </tr>
            ) : (
              items.map((threat) => (
                <tr
                  key={threat.id}
                  className={styles.row}
                  onClick={() => setSelectedThreat(alertToThreatEvent(threat))}
                >
                  <td className={styles.timeCell}>
                    {formatTime(threat.timestamp)}
                  </td>
                  <td className={styles.monoCell}
                   onClick={(e) => {
                    e.stopPropagation()
                    setSourceIp(threat.source_ip)
                    setOffset(0)
                  }}
                  style={{ cursor: 'pointer', color: '#60a5fa' }}
                  >
                    {threat.source_ip}
                  </td>

                  <td>
                    <MethodBadge method={threat.request_method} />
                  </td>
                  <td className={styles.pathCell}>{threat.request_path}</td>
                  <td className={styles.scoreCell}>
                    {threat.threat_score.toFixed(3)}
                  </td>
                  <td>
                    <SeverityBadge severity={threat.severity} />
                  </td>
                  <td>{200}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div className={styles.pagination}>
        <span className={styles.paginationInfo}>
          {total === 0 ? 'No results' : `${offset + 1}–${showing} of ${total}`}
        </span>
        <div className={styles.paginationButtons}>
          <button
            type="button"
            className={styles.pageBtn}
            disabled={!hasPrev}
            onClick={() => setOffset(offset - PAGINATION.DEFAULT_LIMIT)}
          >
            Previous
          </button>
          <button
            type="button"
            className={styles.pageBtn}
            disabled={!hasNext}
            onClick={() => setOffset(offset + PAGINATION.DEFAULT_LIMIT)}
          >
            Next
          </button>
        </div>
      </div>

      <ThreatDetail
        threat={selectedThreat}
        onClose={() => setSelectedThreat(null)}
        onNext={handleNext}
        onPrevious={handlePrevious}
      />
    </div>
  )
}

Component.displayName = 'ThreatsPage'
