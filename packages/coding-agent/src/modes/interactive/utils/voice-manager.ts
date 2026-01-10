import type { InteractiveModeContext } from "../types";

const VOICE_PROGRESS_DELAY_MS = 15000;
const VOICE_PROGRESS_MIN_CHARS = 160;
const VOICE_PROGRESS_DELTA_CHARS = 120;

export class VoiceManager {
	constructor(private ctx: InteractiveModeContext) {}

	setVoiceStatus(text: string | undefined): void {
		this.ctx.statusLine.setHookStatus("voice", text);
		this.ctx.ui.requestRender();
	}

	async handleVoiceInterrupt(reason?: string): Promise<void> {
		const now = Date.now();
		if (now - this.ctx.lastVoiceInterruptAt < 200) return;
		this.ctx.lastVoiceInterruptAt = now;
		if (this.ctx.session.isBashRunning) {
			this.ctx.session.abortBash();
		}
		if (this.ctx.session.isStreaming) {
			await this.ctx.session.abort();
		}
		if (reason) {
			this.ctx.showStatus(reason);
		}
	}

	stopVoiceProgressTimer(): void {
		if (this.ctx.voiceProgressTimer) {
			clearTimeout(this.ctx.voiceProgressTimer);
			this.ctx.voiceProgressTimer = undefined;
		}
	}

	startVoiceProgressTimer(): void {
		this.stopVoiceProgressTimer();
		if (!this.ctx.settingsManager.getVoiceEnabled() || !this.ctx.voiceAutoModeEnabled) return;
		this.ctx.voiceProgressSpoken = false;
		this.ctx.voiceProgressLastLength = 0;
		this.ctx.voiceProgressTimer = setTimeout(() => {
			void this.maybeSpeakProgress();
		}, VOICE_PROGRESS_DELAY_MS);
	}

	async maybeSpeakProgress(): Promise<void> {
		if (!this.ctx.session.isStreaming || this.ctx.voiceProgressSpoken || !this.ctx.voiceAutoModeEnabled) return;
		const streaming = this.ctx.streamingMessage;
		if (!streaming) return;
		const text = this.ctx.extractAssistantText(streaming);
		if (!text || text.length < VOICE_PROGRESS_MIN_CHARS) {
			if (this.ctx.session.isStreaming) {
				this.ctx.voiceProgressTimer = setTimeout(() => {
					void this.maybeSpeakProgress();
				}, VOICE_PROGRESS_DELAY_MS);
			}
			return;
		}

		const delta = text.length - this.ctx.voiceProgressLastLength;
		if (delta < VOICE_PROGRESS_DELTA_CHARS) {
			if (this.ctx.session.isStreaming) {
				this.ctx.voiceProgressTimer = setTimeout(() => {
					void this.maybeSpeakProgress();
				}, VOICE_PROGRESS_DELAY_MS);
			}
			return;
		}

		this.ctx.voiceProgressLastLength = text.length;
		this.ctx.voiceProgressSpoken = true;
		this.ctx.voiceSupervisor.notifyProgress(text);
	}

	async submitVoiceText(text: string): Promise<void> {
		const cleaned = text.trim();
		if (!cleaned) {
			this.ctx.showWarning("No speech detected. Try again.");
			return;
		}
		const toSend = cleaned;
		this.ctx.editor.addToHistory(toSend);

		if (this.ctx.session.isStreaming) {
			await this.ctx.session.abort();
			await this.ctx.session.steer(toSend);
			this.ctx.updatePendingMessagesDisplay();
			return;
		}

		if (this.ctx.onInputCallback) {
			this.ctx.onInputCallback({ text: toSend });
		}
	}
}
