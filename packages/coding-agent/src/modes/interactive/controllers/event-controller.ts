import { Loader, Text } from "@oh-my-pi/pi-tui";
import type { AgentSessionEvent } from "../../../core/agent-session";
import { detectNotificationProtocol, isNotificationSuppressed, sendNotification } from "../../../core/terminal-notify";
import { AssistantMessageComponent } from "../components/assistant-message";
import { ToolExecutionComponent } from "../components/tool-execution";
import { TtsrNotificationComponent } from "../components/ttsr-notification";
import { getSymbolTheme, theme } from "../theme/theme";
import type { InteractiveModeContext } from "../types";

export class EventController {
	constructor(private ctx: InteractiveModeContext) {}

	subscribeToAgent(): void {
		this.ctx.unsubscribe = this.ctx.session.subscribe(async (event: AgentSessionEvent) => {
			await this.handleEvent(event);
		});
	}

	async handleEvent(event: AgentSessionEvent): Promise<void> {
		if (!this.ctx.isInitialized) {
			await this.ctx.init();
		}

		this.ctx.statusLine.invalidate();
		this.ctx.updateEditorTopBorder();

		switch (event.type) {
			case "agent_start":
				if (this.ctx.retryEscapeHandler) {
					this.ctx.editor.onEscape = this.ctx.retryEscapeHandler;
					this.ctx.retryEscapeHandler = undefined;
				}
				if (this.ctx.retryLoader) {
					this.ctx.retryLoader.stop();
					this.ctx.retryLoader = undefined;
					this.ctx.statusContainer.clear();
				}
				if (this.ctx.loadingAnimation) {
					this.ctx.loadingAnimation.stop();
				}
				this.ctx.statusContainer.clear();
				this.ctx.loadingAnimation = new Loader(
					this.ctx.ui,
					(spinner) => theme.fg("accent", spinner),
					(text) => theme.fg("muted", text),
					`Working${theme.format.ellipsis} (esc to interrupt)`,
					getSymbolTheme().spinnerFrames,
				);
				this.ctx.statusContainer.addChild(this.ctx.loadingAnimation);
				this.ctx.startVoiceProgressTimer();
				this.ctx.ui.requestRender();
				break;

			case "message_start":
				if (event.message.role === "hookMessage" || event.message.role === "custom") {
					this.ctx.addMessageToChat(event.message);
					this.ctx.ui.requestRender();
				} else if (event.message.role === "user") {
					this.ctx.addMessageToChat(event.message);
					this.ctx.editor.setText("");
					this.ctx.updatePendingMessagesDisplay();
					this.ctx.ui.requestRender();
				} else if (event.message.role === "fileMention") {
					this.ctx.addMessageToChat(event.message);
					this.ctx.ui.requestRender();
				} else if (event.message.role === "assistant") {
					this.ctx.streamingComponent = new AssistantMessageComponent(undefined, this.ctx.hideThinkingBlock);
					this.ctx.streamingMessage = event.message;
					this.ctx.chatContainer.addChild(this.ctx.streamingComponent);
					this.ctx.streamingComponent.updateContent(this.ctx.streamingMessage);
					this.ctx.ui.requestRender();
				}
				break;

			case "message_update":
				if (this.ctx.streamingComponent && event.message.role === "assistant") {
					this.ctx.streamingMessage = event.message;
					this.ctx.streamingComponent.updateContent(this.ctx.streamingMessage);

					for (const content of this.ctx.streamingMessage.content) {
						if (content.type === "toolCall") {
							if (!this.ctx.pendingTools.has(content.id)) {
								this.ctx.chatContainer.addChild(new Text("", 0, 0));
								const tool = this.ctx.session.getToolByName(content.name);
								const component = new ToolExecutionComponent(
									content.name,
									content.arguments,
									{
										showImages: this.ctx.settingsManager.getShowImages(),
									},
									tool,
									this.ctx.ui,
									this.ctx.sessionManager.getCwd(),
								);
								component.setExpanded(this.ctx.toolOutputExpanded);
								this.ctx.chatContainer.addChild(component);
								this.ctx.pendingTools.set(content.id, component);
							} else {
								const component = this.ctx.pendingTools.get(content.id);
								if (component) {
									component.updateArgs(content.arguments);
								}
							}
						}
					}
					this.ctx.ui.requestRender();
				}
				break;

			case "message_end":
				if (event.message.role === "user") break;
				if (this.ctx.streamingComponent && event.message.role === "assistant") {
					this.ctx.streamingMessage = event.message;
					if (this.ctx.session.isTtsrAbortPending && this.ctx.streamingMessage.stopReason === "aborted") {
						const msgWithoutAbort = { ...this.ctx.streamingMessage, stopReason: "stop" as const };
						this.ctx.streamingComponent.updateContent(msgWithoutAbort);
					} else {
						this.ctx.streamingComponent.updateContent(this.ctx.streamingMessage);
					}

					if (
						this.ctx.streamingMessage.stopReason === "aborted" ||
						this.ctx.streamingMessage.stopReason === "error"
					) {
						if (!this.ctx.session.isTtsrAbortPending) {
							let errorMessage: string;
							if (this.ctx.streamingMessage.stopReason === "aborted") {
								const retryAttempt = this.ctx.session.retryAttempt;
								errorMessage =
									retryAttempt > 0
										? `Aborted after ${retryAttempt} retry attempt${retryAttempt > 1 ? "s" : ""}`
										: "Operation aborted";
							} else {
								errorMessage = this.ctx.streamingMessage.errorMessage || "Error";
							}
							for (const [, component] of this.ctx.pendingTools.entries()) {
								component.updateResult({
									content: [{ type: "text", text: errorMessage }],
									isError: true,
								});
							}
						}
						this.ctx.pendingTools.clear();
					} else {
						for (const [, component] of this.ctx.pendingTools.entries()) {
							component.setArgsComplete();
						}
					}
					this.ctx.streamingComponent = undefined;
					this.ctx.streamingMessage = undefined;
					this.ctx.statusLine.invalidate();
					this.ctx.updateEditorTopBorder();
				}
				this.ctx.ui.requestRender();
				break;

			case "tool_execution_start": {
				if (!this.ctx.pendingTools.has(event.toolCallId)) {
					const tool = this.ctx.session.getToolByName(event.toolName);
					const component = new ToolExecutionComponent(
						event.toolName,
						event.args,
						{
							showImages: this.ctx.settingsManager.getShowImages(),
						},
						tool,
						this.ctx.ui,
						this.ctx.sessionManager.getCwd(),
					);
					component.setExpanded(this.ctx.toolOutputExpanded);
					this.ctx.chatContainer.addChild(component);
					this.ctx.pendingTools.set(event.toolCallId, component);
					this.ctx.ui.requestRender();
				}
				break;
			}

			case "tool_execution_update": {
				const component = this.ctx.pendingTools.get(event.toolCallId);
				if (component) {
					component.updateResult({ ...event.partialResult, isError: false }, true);
					this.ctx.ui.requestRender();
				}
				break;
			}

			case "tool_execution_end": {
				const component = this.ctx.pendingTools.get(event.toolCallId);
				if (component) {
					component.updateResult({ ...event.result, isError: event.isError });
					this.ctx.pendingTools.delete(event.toolCallId);
					this.ctx.ui.requestRender();
				}
				break;
			}

			case "agent_end":
				this.ctx.stopVoiceProgressTimer();
				if (this.ctx.loadingAnimation) {
					this.ctx.loadingAnimation.stop();
					this.ctx.loadingAnimation = undefined;
					this.ctx.statusContainer.clear();
				}
				if (this.ctx.streamingComponent) {
					this.ctx.chatContainer.removeChild(this.ctx.streamingComponent);
					this.ctx.streamingComponent = undefined;
					this.ctx.streamingMessage = undefined;
				}
				this.ctx.pendingTools.clear();
				if (this.ctx.settingsManager.getVoiceEnabled() && this.ctx.voiceAutoModeEnabled) {
					const lastAssistant = this.ctx.findLastAssistantMessage();
					if (lastAssistant && lastAssistant.stopReason !== "aborted" && lastAssistant.stopReason !== "error") {
						const text = this.ctx.extractAssistantText(lastAssistant);
						if (text) {
							this.ctx.voiceSupervisor.notifyResult(text);
						}
					}
				}
				this.ctx.ui.requestRender();
				this.sendCompletionNotification();
				break;

			case "auto_compaction_start": {
				this.ctx.autoCompactionEscapeHandler = this.ctx.editor.onEscape;
				this.ctx.editor.onEscape = () => {
					this.ctx.session.abortCompaction();
				};
				this.ctx.statusContainer.clear();
				const reasonText = event.reason === "overflow" ? "Context overflow detected, " : "";
				this.ctx.autoCompactionLoader = new Loader(
					this.ctx.ui,
					(spinner) => theme.fg("accent", spinner),
					(text) => theme.fg("muted", text),
					`${reasonText}Auto-compacting${theme.format.ellipsis} (esc to cancel)`,
					getSymbolTheme().spinnerFrames,
				);
				this.ctx.statusContainer.addChild(this.ctx.autoCompactionLoader);
				this.ctx.ui.requestRender();
				break;
			}

			case "auto_compaction_end": {
				if (this.ctx.autoCompactionEscapeHandler) {
					this.ctx.editor.onEscape = this.ctx.autoCompactionEscapeHandler;
					this.ctx.autoCompactionEscapeHandler = undefined;
				}
				if (this.ctx.autoCompactionLoader) {
					this.ctx.autoCompactionLoader.stop();
					this.ctx.autoCompactionLoader = undefined;
					this.ctx.statusContainer.clear();
				}
				if (event.aborted) {
					this.ctx.showStatus("Auto-compaction cancelled");
				} else if (event.result) {
					this.ctx.chatContainer.clear();
					this.ctx.rebuildChatFromMessages();
					this.ctx.addMessageToChat({
						role: "compactionSummary",
						tokensBefore: event.result.tokensBefore,
						summary: event.result.summary,
						timestamp: Date.now(),
					});
					this.ctx.statusLine.invalidate();
					this.ctx.updateEditorTopBorder();
				}
				await this.ctx.flushCompactionQueue({ willRetry: event.willRetry });
				this.ctx.ui.requestRender();
				break;
			}

			case "auto_retry_start": {
				this.ctx.retryEscapeHandler = this.ctx.editor.onEscape;
				this.ctx.editor.onEscape = () => {
					this.ctx.session.abortRetry();
				};
				this.ctx.statusContainer.clear();
				const delaySeconds = Math.round(event.delayMs / 1000);
				this.ctx.retryLoader = new Loader(
					this.ctx.ui,
					(spinner) => theme.fg("warning", spinner),
					(text) => theme.fg("muted", text),
					`Retrying (${event.attempt}/${event.maxAttempts}) in ${delaySeconds}s${theme.format.ellipsis} (esc to cancel)`,
					getSymbolTheme().spinnerFrames,
				);
				this.ctx.statusContainer.addChild(this.ctx.retryLoader);
				this.ctx.ui.requestRender();
				break;
			}

			case "auto_retry_end": {
				if (this.ctx.retryEscapeHandler) {
					this.ctx.editor.onEscape = this.ctx.retryEscapeHandler;
					this.ctx.retryEscapeHandler = undefined;
				}
				if (this.ctx.retryLoader) {
					this.ctx.retryLoader.stop();
					this.ctx.retryLoader = undefined;
					this.ctx.statusContainer.clear();
				}
				if (!event.success) {
					this.ctx.showError(
						`Retry failed after ${event.attempt} attempts: ${event.finalError || "Unknown error"}`,
					);
				}
				this.ctx.ui.requestRender();
				break;
			}

			case "ttsr_triggered": {
				const component = new TtsrNotificationComponent(event.rules);
				component.setExpanded(this.ctx.toolOutputExpanded);
				this.ctx.chatContainer.addChild(component);
				this.ctx.ui.requestRender();
				break;
			}
		}
	}

	sendCompletionNotification(): void {
		if (this.ctx.isBackgrounded === false) return;
		if (isNotificationSuppressed()) return;
		const method = this.ctx.settingsManager.getNotificationOnComplete();
		if (method === "off") return;
		const protocol = method === "auto" ? detectNotificationProtocol() : method;
		const title = this.ctx.sessionManager.getSessionTitle();
		const message = title ? `${title}: Complete` : "Complete";
		sendNotification(protocol, message);
	}

	async handleBackgroundEvent(event: AgentSessionEvent): Promise<void> {
		if (event.type !== "agent_end") {
			return;
		}
		if (this.ctx.session.queuedMessageCount > 0 || this.ctx.session.isStreaming) {
			return;
		}
		this.sendCompletionNotification();
		await this.ctx.shutdown();
	}
}
