import { expect, test } from '@playwright/test';

const demoPath = `/webgpu_app/demo/index.html${process.env.DEMO_QUERY ?? ''}`;

test('world model demo loads its stylesheet @demo', async ({ page }) => {
  await page.goto(demoPath);
  await expect
    .poll(async () =>
      page.locator('.shell').evaluate((element) => getComputedStyle(element).display),
    )
    .toBe('grid');
  await expect
    .poll(async () =>
      page.locator('.machine').evaluate((element) => getComputedStyle(element).borderTopStyle),
    )
    .toBe('solid');
});

test('world model demo starts and renders a frame @demo', async ({ page }) => {
  await page.goto(demoPath);
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });
  await page.locator('#start').click();
  await expect
    .poll(async () => Number(await page.locator('#frame-count').textContent()), {
      timeout: 180_000,
    })
    .toBeGreaterThan(0);
  await expect(page.locator('#latency')).not.toHaveText('-- ms');
});

test('world model demo changes the canvas over generated frames @demo', async ({ page }) => {
  await page.goto(demoPath);
  await expect(page.locator('#status')).toContainText('Ready', { timeout: 180_000 });
  await page.locator('#start').click();

  const samples = [];
  for (const targetFrame of [1, 2, 3, 4, 5, 8, 12]) {
    await expect
      .poll(async () => Number(await page.locator('#frame-count').textContent()), {
        timeout: 180_000,
      })
      .toBeGreaterThanOrEqual(targetFrame);
    samples.push(
      await page.locator('#frame').evaluate((canvas) => {
        const context = canvas.getContext('2d');
        const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
        let hash = 2166136261;
        for (let index = 0; index < data.length; index += 4) {
          hash ^= data[index] + (data[index + 1] << 8) + (data[index + 2] << 16);
          hash = Math.imul(hash, 16777619);
        }
        return hash >>> 0;
      }),
    );
  }

  expect(new Set(samples).size).toBeGreaterThan(1);
});
