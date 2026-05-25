---
title: YT Downloader
draft: false
tags:
  - digital_tools   
  - apps
  - productivity
  - utility
  - yt
date: 2024-03-16 23:34
---
# YT-Playlist Download using yt-dlp cli tool

Learn how to download a playlist with subtitles to a new folder named after the playlist using yt-dlp with the best video and audio encoded together:

1. **Install yt-dlp**: Ensure that you have yt-dlp installed and configured on your system. For more information on installation, refer to the [yt-dlp GitHub repository](https://github.com/yt-dlp/yt-dlp).

2. **Create the command**: Use the following command with the appropriate playlist ID:

    ```bash
    yt-dlp -o "%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s" --write-subs --sub-langs all --format "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]" "https://www.youtube.com/playlist?list=PLAYLIST_ID"
    ```

3. **Understand the command**:

   - -o "%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s": Specifies the output format and creates a new folder named after the playlist, with files named by their index and title in the playlist.
   - --write-subs: Enables subtitle downloading.
   - --sub-langs all: Downloads subtitles in all available languages.
   - --format "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]": Sets the format to download the best video and audio encoded together.

4. Execute the command: Replace PLAYLIST_ID with the actual playlist ID from YouTube and run the command in your terminal or command prompt.

By following these steps, you'll be able to download the desired playlist with subtitles into a new folder named after the playlist, with the best video and audio quality combined

---

**Additional Commands**
Here are some additional commands for different use cases:

- Best quality commands:

  - No folder:

      ```bash
      yt-dlp.exe -f best/bestvideo+bestaudio  -ciw -o "%(title)s.%(ext)s" --write-auto-sub  -v
      ```

  - With folder:

      ```bash
      yt-dlp.exe -o "%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s" -f best/bestvideo+bestaudio --write-auto-sub -v
      ```

  - 720p playlist with folder:

      ```bash
      yt-dlp.exe -o "%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s" -f "best[height=720]+ba" --write-auto-sub -v
      ```

  - Subtitles only:

    ```bash
    yt-dlp.exe --write-auto-sub --skip-download  -o "%(title)s.%(ext)s" -v  
    yt-dlp.exe --write-sub --skip-download --sub-lang en --convert-subs srt -v
    ```

---
