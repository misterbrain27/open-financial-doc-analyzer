import { test, expect } from '@playwright/test';
import { mockIngest, mockIngestError, mockQuery, mockQueryError } from './fixtures/api-mocks';

test('la page charge', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/FinancialRAG/);
  await expect(page.getByRole('link', { name: 'Upload' })).toBeVisible();
});

test.describe('Chat — question / réponse (SSE mocké)', () => {
  // Double-quoted strings → French apostrophes don't break the JS literal.

  test('affiche la réponse streamée et les sources citées', async ({ page }) => {
    // The mock MUST be set up before navigation: the route must exist by the time the request fires.
    await mockQuery(page, {
      answer: "Le résultat d'exploitation est de 51 750 €.",
      sources: [
        {
          index: 1,
          company: 'novatech',
          year: 2025,
          page_number: 3,
          text: "Résultat d'exploitation : 51 750",
        },
      ],
    });

    await page.goto('/chat');

    const input = page.getByRole('textbox', { name: 'Votre question' });
    await input.fill("Quel est le résultat d'exploitation ?");
    await input.press('Enter');

    await expect(page.getByText("Le résultat d'exploitation est de 51 750 €.")).toBeVisible();
    await expect(page.locator('.sources__list .source')).toHaveCount(1);
    await expect(page.getByText('page 3')).toBeVisible();
  });

  test("affiche un message d'erreur si la génération échoue en cours de flux", async ({ page }) => {
    await mockQueryError(page, {
      message: 'Le service de génération est indisponible.',
      partialAnswer: 'Le résultat',
      sources: [
        { index: 1, company: 'novatech', year: 2025, page_number: 3, text: 'Compte de résultat' },
      ],
    });

    await page.goto('/chat');

    const input = page.getByRole('textbox', { name: 'Votre question' });
    await input.fill("Quel est le résultat d'exploitation ?");
    await input.press('Enter');

    await expect(page.locator('.bubble__error')).toContainText(
      'Le service de génération est indisponible.',
    );
  });

  test("n'affiche pas de panneau Sources quand le retrieval ne renvoie rien", async ({ page }) => {
    await mockQuery(page, {
      answer: "Je n'ai pas trouvé d'information pertinente dans les documents.",
      sources: [],
    });

    await page.goto('/chat');

    const input = page.getByRole('textbox', { name: 'Votre question' });
    await input.fill('Quel est le cours de bourse ?');
    await input.press('Enter');

    await expect(
      page.getByText("Je n'ai pas trouvé d'information pertinente dans les documents."),
    ).toBeVisible();
    await expect(page.locator('.sources__list')).toHaveCount(0);
  });
});

test.describe('Upload — ingestion (POST /ingest mocké)', () => {
  // In-memory file: the backend is mocked, only the name (.pdf) is checked on the frontend.
  const pdfFile = {
    name: 'rapport.pdf',
    mimeType: 'application/pdf',
    buffer: Buffer.from('%PDF-1.4\nmock'),
  };

  test('ingère un PDF et affiche la carte de succès', async ({ page }) => {
    await mockIngest(page, {
      id: 1,
      source_path: 'novatech_2025.pdf',
      company: 'novatech',
      year: 2025,
    });

    await page.goto('/');

    // The file input is hidden: setInputFiles targets it directly and triggers the `change` event.
    await page.locator('input[type="file"]').setInputFiles(pdfFile);
    await page.getByRole('button', { name: /Lancer/ }).click();

    const successCard = page.locator('.card--success');
    await expect(successCard).toContainText('Document ingéré');
    await expect(successCard).toContainText('novatech_2025.pdf');
  });

  test("affiche une erreur si l'ingestion échoue", async ({ page }) => {
    await mockIngestError(page, { status: 400, detail: 'Le PDF est illisible ou corrompu.' });

    await page.goto('/');

    await page.locator('input[type="file"]').setInputFiles(pdfFile);
    await page.getByRole('button', { name: /Lancer/ }).click();

    const errorCard = page.locator('.card--error');
    await expect(errorCard).toContainText('Échec');
    await expect(errorCard).toContainText('Le PDF est illisible ou corrompu.');
  });
});
