// ===================
// ThreatShield AI | 2026
// index.tsx
// ===================

import { useAlerts, useModelStatus, useStats } from '@/api/hooks'
import { AlertFeed, StatCard } from '@/components'
import styles from './dashboard.module.scss'

function SeverityBar({
  high,
  medium,
  low,
}: {
  high: number
  medium: number
  low: number
}): React.ReactElement {
  const total = high + medium + low
  if (total === 0) {
    return <div className={styles.severityBarEmpty}>No threats detected</div>
  }

  return (
    <div className={styles.severityBar}>
      <div
        className={styles.severityHigh}
        style={{ width: `${(high / total) * 100}%` }}
      />
      <div
        className={styles.severityMedium}
        style={{ width: `${(medium / total) * 100}%` }}
      />
      <div
        className={styles.severityLow}
        style={{ width: `${(low / total) * 100}%` }}
      />
    </div>
  )
}

function SeverityLegend({
  high,
  medium,
  low,
}: {
  high: number
  medium: number
  low: number
}): React.ReactElement {
  return (
    <div className={styles.legend}>
      <span className={styles.legendItem}>
        <span className={`${styles.legendDot} ${styles.dotHigh}`} />
        High: {high}
      </span>
      <span className={styles.legendItem}>
        <span className={`${styles.legendDot} ${styles.dotMedium}`} />
        Medium: {medium}
      </span>
      <span className={styles.legendItem}>
        <span className={`${styles.legendDot} ${styles.dotLow}`} />
        Low: {low}
      </span>
    </div>
  )
}

function RankedList({
  title,
  items,
}: {
  title: string
  items: { label: string; count: number }[]
}): React.ReactElement {
  return (
    <div className={styles.rankedList}>
      <h3 className={styles.rankedTitle}>{title}</h3>
      {items.length === 0 ? (
        <span className={styles.emptyText}>None</span>
      ) : (
        <ol className={styles.rankedItems}>
          {items.map((item) => (
            <li key={item.label} className={styles.rankedItem}>
              <span className={styles.rankedLabel}>{item.label}</span>
              <span className={styles.rankedCount}>{item.count}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

export function Component(): React.ReactElement {
  if (!localStorage.getItem('auth')) {
    window.location.href = '/login'
  }
  const handleLogout = (): void => {
    localStorage.removeItem('auth')
    window.location.href = '/login'
  }
  const { data: stats, isLoading: statsLoading } = useStats()
  const { data: modelStatus } = useModelStatus()
  const { alerts, isConnected, connectionError } = useAlerts()

  if (statsLoading || !stats) {
    return <div className={styles.loading}>Loading dashboard...</div>
  }

  const high = alerts.filter((a) => a.severity === 'HIGH').length
  const medium = alerts.filter((a) => a.severity === 'MEDIUM').length
  const low = alerts.filter((a) => a.severity === 'LOW').length
  const totalThreats = alerts.length
  const topIpsMap: Record<string, number> = {}
  alerts.forEach((alert) => {
    topIpsMap[alert.source_ip] =
    (topIpsMap[alert.source_ip] || 0) + 1
  })
 const topIps = Object.entries(topIpsMap)
  .map(([label, count]) => ({ label, count }))
  .sort((a, b) => b.count - a.count)
  .slice(0, 5)
  
  const topPathsMap: Record<string, number> = {}
  alerts.forEach((alert) => {
    topPathsMap[alert.request_path] =
    (topPathsMap[alert.request_path] || 0) + 1
  })
  const topPaths = Object.entries(topPathsMap).map(([label, count]) => ({
    label,
    count,
  }))
  .sort((a, b) => b.count - a.count)
  .slice(0, 5)

  return (
    <div className={styles.page}>
       <div className={styles.topBar}>
        <button className={styles.logoutButton} onClick={handleLogout}>
          Logout
        </button>
       </div>
      <div className={styles.statRow}>
        <StatCard label="Threats Detected" value={totalThreats} />
        <StatCard label="Threats Stored" value={totalThreats} />
        <StatCard
          label="high severity"
          value={high}
          sublabel={`of ${totalThreats} total`}
        />
        <StatCard
          label="Detection Mode"
          value={modelStatus?.detection_mode ?? '...'}
          sublabel={modelStatus?.models_loaded ? 'Models loaded' : 'Rules only'}
        />
      </div>

      <div className={styles.severitySection}>
        <SeverityBar high={high} medium={medium} low={low} />
        <SeverityLegend high={high} medium={medium} low={low} />
      </div>

      <div className={styles.bottomRow}>
        <AlertFeed alerts={alerts} isConnected={isConnected} maxHeight="360px" />

        <div className={styles.lists}>
          <RankedList
            title="Top Source IPs"
            items={topIps}
          />
          <RankedList
            title="Top Attacked Paths"
            items={topPaths}
          />
        </div>
      </div>

      {connectionError && <div className={styles.wsError}>{connectionError}</div>}
    </div>
  )
}

Component.displayName = 'DashboardPage'
