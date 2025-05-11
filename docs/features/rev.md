# Rev.AI
Batchalign supports ASR transcription via the [Rev.AI](https://www.rev.ai/) service, by default via the standard `batchalign transcribe` command. To use it, you must setup a Rev.AI key.

## Obtaining a Rev.AI Key

To run transcribe using Rev-AI, you can sign up for a free account at https://www.rev.ai/auth/signup. Then you sign in and you can get your free access token at https://www.rev.ai/access_token. Copy the token and enter it into a file called rev_key.  During installation, Batchalign will prompt you to enter this key, so you can use ASR from Rev-AI for 5 free hours of audio.  Later you can decide whether or not to switch to a paid subscription using the same key.

## Privacy
If your IRB protocols have privacy concerns, note that when you set up an account with Rev-AI, you can select the option to “never store media”.

They state that "Rev AI's Automatic Speech Recognition (ASR) and associated API enable covered entities and business associates to utilize Rev AI services in a manner that complies with the Health Insurance Portability and Accountability Act (HIPAA)."
