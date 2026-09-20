import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  use: {
    baseURL: 'http://localhost:4200',
    // Trace rejouable (timeline + DOM + réseau) capturée uniquement lors d'un rejeu après échec.
    trace: 'on-first-retry',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'npm run start',
    url: 'http://localhost:4200',
    // Réutilise le dev server déjà lancé en local ; le (re)lance en CI.
    reuseExistingServer: !process.env.CI,
    // Le premier build Angular à froid peut dépasser le défaut de 60 s.
    timeout: 120 * 1000,
  },
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
});
