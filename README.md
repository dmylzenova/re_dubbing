# re_dubbing

Almost Seamless Audio and Video Dubbing without Optional Lip-sync.

---


1. **Clone the Repository and install Requirements**  
   ```bash
   git clone https://github.com/dmylzenova/re_dubbing.git
   cd repo
 
   sh setup.sh
   conda create -n redub python=3.10
   conda activate redub
   pip install -e .
   ```

2.  **Export Huggingface and Elevenlabs access keys**
    ```bash
    export ELEVENLABS_API_KEY=YOUR_11LABS_KEY
    export HUGGINGFACE_ACCESS_TOKEN=YOUR_HUGGINGFACE_TOKEN
    ```

3. **Run main.py providing paths to video, original and edited transcriptions.**
   ```bash
   python main.py -v ../1/video.mp4  -o ../1/original_transcription.srt -e ../1/edited_transription.srt
   ```


There are a few assumptions I made, and I didn’t address their handling:

    - No background music (if there is, source separation can be added).
    - Original transcriptions and edited transcriptions align and don’t modify timestamps.

