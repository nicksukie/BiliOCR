# BiliOCR - Real-time translation of embedded video subtitles. 

## What is it? 
 
BiliOCR is a screen reader tool for Mac that can translate subtitles in real time. It allows you to select an area of your screen with subtitles in one language, and the subtitles will be translated to your target language and displayed elegantly on the screen. 

It also has a text-to-speech option to read out the translated subtitles for a dubbing effect. 

<img width="1145" height="714" alt="Pasted image 20260222172655" src="https://github.com/user-attachments/assets/398365e5-8af8-443c-9bc4-e6e1eb4e1146" />
<img width="956" height="534" alt="Pasted image 20260222151501" src="https://github.com/user-attachments/assets/626dcd21-f4b3-455f-8dfb-a761ee315ccb" />

## Why is it? 

The inspiration for creating this tool is to provide a simple way to watch and understand videos on Bilibili for people whose native language is not Chinese.
Most videos on Bilibili have subtitles, but they are embedded into the video itself, rather than overlaid, so browser-based subtitle translators don't work.

Other options include audio-based translators, like [realtime-subtitle](https://github.com/Vanyoo/realtime-subtitle/tree/master). 

## Features

### Real-Time Subtitle Translation
- The core feature is the subtitle streaming feature, which uses OCR to read subtitles in the source language, detects complete phrases/sentences, sends for translation, and then displays the translations on the screen with minimal latency
- OCR subtitle translation is ideal for videos where the subtitles are "baked" into the video, and therefore not readable by browser-based translation tools

### Learn Mode - Video watching as a language improvement method. 
- Only available for Chinese source language
- Check off the Learn Mode box at the bottom of the main menu to activate learn mode. 
- When translations begin to populate the subtitle box, keywords will pop up in a separate panel, including the Chinese word, the pinyin pronunciation, and the target language definition. If target language is English, it will first search CEdict for definitions. For other target languages, it will obtain definitions through translation based on the selected model (highly recommend using MT for this, as LLM is overkill). 
- Learn Mode keywords can be starred, which saves them to a local database so you can review them later. Words can also be copied (one or more by highlighting them), or saved into markdown files (full list).

<img width="466" height="525" alt="Pasted image 20260222173311" src="https://github.com/user-attachments/assets/daf0368a-765a-4255-a62f-c1c4191a8bab" />

### Text-to-speech (TTS) - Real-time overdub.
- OCR translation mode comes complete with TTS integration, to achieve an automatic overdub effect.
- There are three TTS speech model options: 
	- Piper, a lightweight local solution (trade-off, speech is not entirely human-like or natural). Piper itself and each speech model downloads automatically the first time it is used.
	- OpenAI-TTS (cloud-based): while is has extremely realistic voices, the latency is relatively high, causing subtitles to outpace the spoken output
	- Elevenlabs (cloud-based): also extremely realistic voices. The flash model has reasonable latency, but it only available for English (other target language will have strong American accents), whereas the multilingual model has proper accents for many languages but higher latency, running into the outpacing issue. 

### Audio mode - Use computer audio to transcribe and translate video speech, no OCR needed. 
- This is an adaptation of [realtime-subtitle ](https://github.com/Vanyoo/realtime-subtitle/tree/master) integrated with our subtitle streaming/reconciling algorithm, real-time updating UI, and LLM/MT integrated translation options. Audio mode is not compatible with TTS, because the dubbed audio will confound the audio input for translation, creating an undesirable feedback loop. 
- Setup requires Blackhole to route computer audio back as input to be detected by the system. See  [realtime-subtitle ](https://github.com/Vanyoo/realtime-subtitle/tree/master) for detailed setup instructions and documentation. 


## Installation
#### Method 1 (Easiest)
1. Click on "Releases" on the right side of this page.
2. Download the BiliOCR.dmg file.
3. Double click the BiliOCR.dmg file
4. Drag the app icon to your "Applications" folder
5. Go to your "Applications" folder →  right-click on the app → Open 
6. Confirm Open

*  The first time you run the app, it will prompt you to input API keys. Only one (LLM or MT) provider key is required, but it is highly recommended to add one of each (i.e. OpenAI + Google Translate). 

* The first time you run the app, you will also be prompted to provide "Screen Recording" permissions. This is required for OCR functionality to work. MacOS will present you with a pop-up which will take you to your settings, and there you simply have to toggle the setting to "on" for the app "BiliOCR". 


#### Method 2 (Advanced)
1. Download the full repo from Github.
2. Install all dependencies (both those in requirements.txt, as well as in Brewfile.)
3. Run the app using your terminal.



## How to Use
- First, obtain and input an API key for *at least* **one** small model (MT) and **one** large model (LLM) provider. Small model translation quality is inferior but good to have as a fallback in case of API outage or latency. Small model translation is also used to catch words that LLM fails to translate, preventing mixed language output. 
- Navigate to Bilibili (or your video player of choice). 
- Select your preferred api provider and model (LLM recommended, and MT used as a fallback).
- Select the OCR Mode tab and choose your preferred OCR model (default is fine for most purposes) 
- For basic subtitle translation, TTS (speech) can be disabled 
- Click OK 
- Drag the red box over the video subtitle area
- Press enter: the box will turn white. This means OCR detection is activated. 
- Press play on your video.
- Subtitles will appear.

### Optional Settings
1. For TTS (speech) mode, using the default voice method, Piper, will cause the model (in addition to the selected voice) to be downloaded locally. OpenAI TTS or Elevenlabs do not download anything but require an API key.
2. Learn Mode may be enabled is the target language is Chinese, for those who wish to study keywords while watching videos


## Usage Notes: 
- While the tool was built with Bilibili in mind, technically it can be used for any Chinese video application, like Tencent video, Youku, Douyin (browser) etc. 
- While the tool was built with Chinese as a source language in mind, technically it can be used with any source language
- As of v1.0, only Chinese as a source language has been tested. We will update this readme with a list of supported language as they are tested and verified functional. Until then, feel free to experiment with various combinations. 
- Certain LLM models translate better on certain source and output languages, depending on their training data. Some models might fail for rare or data-sparse languages. We will also keep a running log of the optimal models for which languages below. The same goes for voice models for TTS.
- While the default settings should work for most Chinese-->English subtitles, different output languages may call for differently tuned settings. This depends on many things including the way languages are tokenized (i.e. word count per unit time), the speed and concentration of words per unit time, concentration of words per source subtitle line, the rate of source subtitle change etc. 

We appreciate community support in helping us to know which language combinations, language-model combinations, and parameter combinations (per language) work well! If you would like to contribute to the community, just enable json outputting (Settings>General) and send us an email summarizing which model(s) you used for which languages along with your json output. Once validated, we will update the running logs below. 

#### Validated Setup Scenarios (Tested language combinations per model with settings parameters) 
- Chinese-->English, All Siliconflow models, OpenAI (GPT 4 models), default settings. 

#### Known bugs
- GPT-5 models seems to be having API response issues. Stick with GPT-4o-mini for now. 
- \[Resolved\] TTS audio might begin to stutter after some time due to competing CPU usage with OCR. Try sending `sudo killall coreaudiod` into your terminal to see if it helps, or refreshing the Format in "Audio Devices">"Macbook Speakers". Otherwise, consider using a Bluetooth audio device. 

### Disclaimer
This app is in no way affiliated with Bilibili 哔哩哔哩.
