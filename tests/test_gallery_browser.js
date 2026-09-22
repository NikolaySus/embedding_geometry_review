async (page) => {
  const results = [];
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  for (const kind of ['2D', '3E', '5A']) {
    for (let panel = 1; panel <= 4; panel++) {
      const id = `${kind}-compact-${panel}`;
      await page.setViewportSize({width: 1280, height: 1000});
      await page.goto(`http://127.0.0.1:8876/interactive/${id}.html`);
      await page.waitForFunction(() => document.querySelector('#az') && document.querySelector('.js-plotly-plot')._fullLayout.scene._scene);
      const initial = await page.evaluate(() => {
        const gd = document.querySelector('.js-plotly-plot');
        return {traces: gd.data.length, points: gd.data[gd.layout.meta.roof_text_index].text.length,
                az: document.querySelector('#az').value, canvas: document.querySelector('canvas').width};
      });
      if (initial.traces !== 28 || initial.points !== 22 || initial.az !== '300' || initial.canvas === 0) throw new Error(`Initial scene: ${id}`);
      await page.locator('#font').fill('9');
      await page.locator('#az').selectOption('60');
      await page.locator('#roof').fill('0.8');
      await page.getByText('Каркас', {exact: true}).click();
      await page.waitForFunction(() => {
        const gd = document.querySelector('.js-plotly-plot'), meta = gd.layout.meta;
        return gd.data[0].visible === false && Math.abs(gd.data[meta.roof_text_index].z[0] - (meta.high + .8 * meta.span)) < 1e-9;
      });
      const downloadPromise = page.waitForEvent('download');
      await page.locator('#save').click();
      const download = await downloadPromise;
      if (download.suggestedFilename() !== `${id}.camera.json`) throw new Error('Download name');
      await download.saveAs(`browser-${id}.camera.json`);
      const state = await page.evaluate(() => {
        const gd = document.querySelector('.js-plotly-plot');
        return {font: gd.data[gd.layout.meta.roof_text_index].textfont.size,
                camera: gd._fullLayout.scene.camera, roof: document.querySelector('#height').textContent};
      });
      if (Math.abs(state.font - 9 * 1000 / (6.2 * 72)) > 1e-9 || state.roof !== '0.80') throw new Error('Control state');
      await page.getByText('Поверхность', {exact: true}).click();
      if (panel === 1) {
        await page.screenshot({path: `browser-${kind}-desktop.png`});
        await page.mouse.move(500, 450);
        await page.mouse.down();
        await page.mouse.move(610, 490, {steps: 12});
        await page.mouse.up();
        await page.waitForTimeout(400);
        const snapped = await page.evaluate(() => {
          const c = document.querySelector('.js-plotly-plot')._fullLayout.scene.camera;
          const angle = (Math.atan2(c.eye.y, c.eye.x) * 180 / Math.PI + 360) % 360;
          return Math.abs(angle / 60 - Math.round(angle / 60)) < 1e-6 && c.projection.type === 'orthographic';
        });
        if (!snapped) throw new Error(`Drag snap: ${id}`);
        await page.setViewportSize({width: 390, height: 844});
        await page.screenshot({path: `browser-${kind}-mobile.png`, fullPage: true});
      }
      results.push({id, initial, controls: 'passed', download: 'passed'});
    }
  }
  if (errors.length) throw new Error(JSON.stringify(errors));
  console.log(JSON.stringify({status: 'passed', panels: results, pageErrors: errors}, null, 2));
}
