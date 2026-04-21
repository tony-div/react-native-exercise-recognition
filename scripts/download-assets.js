const fs = require('fs')
const path = require('path')
const https = require('https')

const ASSETS_DIR = path.join(__dirname, '..', 'example', 'android', 'app', 'src', 'main', 'assets')
const MODEL_URL = 'https://raw.githubusercontent.com/tony-div/WorkoutHacker/main/ai/models/exercise_classifier_rf.json'
const MODEL_NAME = 'exercise_classifier_rf.json'

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    if (fs.existsSync(dest)) {
      console.log(`Skipping ${MODEL_NAME} - already exists`)
      return resolve()
    }
    console.log(`Downloading ${MODEL_NAME}...`)
    const file = fs.createWriteStream(dest)
    https.get(url, (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        return downloadFile(response.headers.location, dest).then(resolve).catch(reject)
      }
      response.pipe(file)
      file.on('finish', () => {
        file.close()
        console.log(`Downloaded ${MODEL_NAME}`)
        resolve()
      })
    }).on('error', (err) => {
      fs.unlink(dest, () => {})
      reject(err)
    })
  })
}

async function main() {
  if (!fs.existsSync(ASSETS_DIR)) {
    fs.mkdirSync(ASSETS_DIR, { recursive: true })
  }
  const dest = path.join(ASSETS_DIR, MODEL_NAME)
  await downloadFile(MODEL_URL, dest)
  console.log('Exercise classifier model downloaded!')
}

main().catch(console.error)