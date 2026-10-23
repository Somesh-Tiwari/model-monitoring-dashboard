const { Builder, By, until } = require('selenium-webdriver');
const chrome = require('selenium-webdriver/chrome');

async function runUITest() {
    const options = new chrome.Options();
    options.addArguments('--headless', '--disable-gpu', '--no-sandbox');
    const driver = await new Builder().forBrowser('chrome').setChromeOptions(options).build();

    try {
        await driver.get('http://localhost:8000/');
        await driver.wait(async () => {
            const name = await driver.findElement(By.id('model-name')).getText();
            return name === 'Wine Classifier';
        }, 10000);

        await driver.findElement(By.id('predict-btn')).click();
        const result = await driver.findElement(By.id('predict-result'));
        await driver.wait(async () => (await result.getText()).startsWith('Predicted '), 10000);
        console.log(`UI prediction succeeded: ${await result.getText()}`);
    } catch (error) {
        console.error(`UI check failed: ${error}`);
        process.exitCode = 1;
    } finally {
        await driver.quit();
    }
}

runUITest();
