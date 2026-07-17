import { useState } from 'react'
import styles from './login.module.css'

export function Component(): React.ReactElement {

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')

  const handleLogin = (): void => {

    if (
      username === 'admin' &&
      password === 'admin123'
    ) {

      localStorage.setItem('auth', 'true')

      window.location.href = '/'

    } else {

      alert('Invalid credentials')
    }
  }

  return (
    <div className={styles.page}>

      <div className={styles.card}>

        <h1 className={styles.title}>
          AI Threat Detection
        </h1>

        <p className={styles.subtitle}>
          Secure Dashboard Login
        </p>

        <input
          className={styles.input}
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <input
          className={styles.input}
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button
          className={styles.button}
          onClick={handleLogin}
        >
          Login
        </button>

      </div>

    </div>
  )
}

Component.displayName = 'LoginPage'
