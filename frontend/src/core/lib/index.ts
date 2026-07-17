import { create } from 'zustand'

interface UIState {
  sidebarOpen: boolean
  sidebarCollapsed: boolean

  toggleSidebar: () => void
  toggleSidebarCollapsed: () => void
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: false,
  sidebarCollapsed: false,

  toggleSidebar: () =>
    set((state) => ({
      sidebarOpen: !state.sidebarOpen,
    })),

  toggleSidebarCollapsed: () =>
    set((state) => ({
      sidebarCollapsed: !state.sidebarCollapsed,
    })),
}))
