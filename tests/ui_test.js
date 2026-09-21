// tests/ui_test.js
const { Builder, By, until } = require('selenium-webdriver');
const chrome = require('selenium-webdriver/chrome');

async function runUITest() {
    // Configure Chrome to run headlessly (without opening a visible window)
    let options = new chrome.Options();
    options.addArguments('--headless');
    options.addArguments('--disable-gpu');
    options.addArguments('--no-sandbox');

    let driver = await new Builder()
        .forBrowser('chrome')
        .setChromeOptions(options)
        .build();

    try {
        console.log("Navigating to dashboard...");
        await driver.get('http://localhost:8000/');

        console.log("Finding the prediction button...");
        let predictBtn = await driver.findElement(By.id('predict-btn'));
        
        console.log("Clicking the button to simulate inference...");
        await predictBtn.click();

        console.log("Waiting for AI response...");
        let resultTextElement = await driver.wait(
            until.elementLocated(By.id('predict-result')), 
            5000
        );
        
        // Wait until the text actually populates
        await driver.wait(async () => {
            const text = await resultTextElement.getText();
            return text.includes('Class:');
        }, 5000);

        let finalOutput = await resultTextElement.getText();
        console.log(`Test Passed! Model responded with: ${finalOutput}`);

    } catch (error) {
        console.error(`Test Failed: ${error}`);
    } finally {
        await driver.quit();
    }
}

runUITest();