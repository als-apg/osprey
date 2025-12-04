/**
 * Shared Execution Plan Viewer Module
 * Used by both Open Web UI functions and documentation
 * Single source of truth for plan visualization and interaction
 */

class ExecutionPlanViewer {
    constructor(config = {}) {
        // Professional Scientific Interface Design Configuration
        this.colors = {
            primary: '#9eb0af',
            primaryHover: '#7a9291',
            secondary: '#95a4b8',
            secondaryHover: '#7a8aa0',
            success: '#9eb0af',
            successHover: '#7a9291',
            danger: '#d66e6e',
            dangerHover: '#c54545',
            warning: '#95a4b8',
            warningHover: '#7a8aa0',
            background: '#ffffff',
            panelBackground: '#f0d3d3',
            border: '#9eb0af',
            borderLight: '#e0e5e5',
            text: '#000000',
            textLight: '#4a4a4a',
            accent: '#95a4b8',
            accentHover: '#7a8aa0'
        };
        
        this.layout = {
            containerMaxWidth: '1400px',
            containerWidth: '95%',
            containerMaxHeight: '90vh',
            containerPadding: '32px',
            containerBorderWidth: '2px',
            headerBottomMargin: '32px',
            headerBottomPadding: '20px',
            headerBorderWidth: '1px',
            controlPanelPadding: '20px',
            controlPanelBottomMargin: '24px',
            controlPanelGap: '24px',
            buttonSpacing: '12px',
            contentGap: '28px',
            sectionHeaderBottomMargin: '16px',
            sectionHeaderBottomPadding: '8px',
            stepsContainerMinHeight: '300px',
            stepsContainerPadding: '20px',
            stepCardBottomMargin: '16px',
            stepCardPadding: '20px',
            stepCardHeaderBottomMargin: '16px',
            stepCardHeaderBottomPadding: '12px',
            stepCardContentGap: '16px',
            capabilitiesPanelMaxHeight: '500px',
            capabilitiesPanelPadding: '20px',
            capabilityCardBottomMargin: '12px',
            capabilityCardPadding: '16px',
            validationStatusPadding: '16px',
            validationStatusBottomMargin: '20px',
            validationStatusBorderWidth: '1px'
        };
        
        this.typography = {
            mainHeaderFontSize: '20px',
            mainHeaderLetterSpacing: '0.5px',
            sectionHeaderFontSize: '16px',
            sectionHeaderLetterSpacing: '0.5px',
            bodyFontSize: '14px',
            bodyLineHeight: '1.5',
            smallTextFontSize: '13px',
            smallTextLineHeight: '1.5',
            labelFontSize: '12px',
            labelLetterSpacing: '0.5px',
            buttonFontSize: '13px',
            buttonLetterSpacing: '0.5px',
            stepButtonFontSize: '11px',
            stepButtonLetterSpacing: '0.5px',
            capabilityTitleFontSize: '13px',
            capabilityTitleLetterSpacing: '0.5px',
            capabilityDescFontSize: '12px',
            capabilityDescLineHeight: '1.4',
            capabilityMetaFontSize: '11px',
            capabilityMetaLineHeight: '1.4',
            stepTitleFontSize: '11px',
            stepTitleLetterSpacing: '0.5px',
            stepCapabilityFontSize: '14px',
            stepCapabilityLetterSpacing: '0.3px',
            badgeFontSize: '11px',
            badgeLetterSpacing: '0.3px',
            validationFontSize: '13px',
            validationLetterSpacing: '0.3px',
            subtitleFontSize: '13px',
            noInputsFontSize: '11px',
            noInputsLetterSpacing: '0.3px'
        };
        
        this.borderRadius = {
            container: '4px',
            panel: '3px',
            button: '3px',
            field: '3px',
            badge: '3px'
        };
        
        this.shadows = {
            container: '0 8px 16px rgba(0, 0, 0, 0.1)',
            card: '0 2px 4px rgba(0,0,0,0.05)',
            cardHover: '0 4px 8px rgba(0,0,0,0.1)',
            modal: '0 25px 50px -12px rgba(0, 0, 0, 0.25)'
        };
        
        this.transitions = {
            default: 'all 0.2s',
            fast: '0.15s',
            slow: '0.3s'
        };
        
        // Override with custom config
        Object.assign(this, config);
        
        // Internal state
        this.currentPlan = [];
        this.stepCounter = 0;
        this.isReviewMode = false;
        this.reviewModeData = null;
        this.capabilities = [];
        this.contextTypes = [];
        this.templates = [];
        this.availableContextKeys = [];
        this.callbacks = {
            onSave: null,
            onCancel: null,
            onValidate: null
        };
    }
    
    /**
     * Initialize the viewer with data and options
     */
    init(options = {}) {
        this.capabilities = options.capabilities || [];
        this.contextTypes = options.contextTypes || [];
        this.templates = options.templates || [];
        this.availableContextKeys = options.availableContextKeys || [];
        this.callbacks = { ...this.callbacks, ...options.callbacks };
        
        // Check if we have pending plan data (review mode)
        if (options.pendingPlan) {
            this.isReviewMode = true;
            this.reviewModeData = {
                originalTask: options.pendingPlan.__metadata__?.original_task || "Unknown task",
                contextKey: options.pendingPlan.__metadata__?.context_key || "unknown",
                createdAt: options.pendingPlan.__metadata__?.created_at || "Unknown time"
            };
            this.currentPlan = options.pendingPlan.steps || [];
            this.stepCounter = this.currentPlan.length;
        }
    }
    
    /**
     * Create the main viewer container
     */
    createContainer(parentElement, options = {}) {
        const container = document.createElement('div');
        const isModal = options.modal !== false; // Default to modal
        
        if (isModal) {
            // Create overlay for modal mode
            const overlay = document.createElement('div');
            overlay.id = 'plan-editor-overlay';
            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                z-index: 10000;
                display: flex;
                justify-content: center;
                align-items: center;
            `;
            
            container.style.cssText = `
                background: ${this.colors.background};
                border: ${this.layout.containerBorderWidth} solid ${this.colors.border};
                border-radius: ${this.borderRadius.container};
                padding: ${this.layout.containerPadding};
                width: ${this.layout.containerWidth};
                max-width: ${this.layout.containerMaxWidth};
                max-height: ${this.layout.containerMaxHeight};
                overflow-y: auto;
                box-shadow: ${this.shadows.container};
                color: ${this.colors.text};
                font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                font-size: ${this.typography.bodyFontSize};
                line-height: ${this.typography.bodyLineHeight};
            `;
            
            overlay.appendChild(container);
            parentElement.appendChild(overlay);
            
            // Close on overlay click
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay && this.callbacks.onCancel) {
                    this.callbacks.onCancel();
                }
            });
            
            return { container, overlay };
        } else {
            // Inline mode for documentation
            container.style.cssText = `
                background: ${this.colors.background};
                border: ${this.layout.containerBorderWidth} solid ${this.colors.border};
                border-radius: ${this.borderRadius.container};
                padding: ${this.layout.containerPadding};
                width: 100%;
                max-width: ${this.layout.containerMaxWidth};
                box-shadow: ${this.shadows.container};
                color: ${this.colors.text};
                font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                font-size: ${this.typography.bodyFontSize};
                line-height: ${this.typography.bodyLineHeight};
                margin: 20px 0;
            `;
            
            parentElement.appendChild(container);
            return { container };
        }
    }
    
    /**
     * Create header section
     */
    createHeader() {
        const header = document.createElement('div');
        header.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: ${this.layout.headerBottomMargin};
            padding-bottom: ${this.layout.headerBottomPadding};
            border-bottom: ${this.layout.headerBorderWidth} solid ${this.colors.border};
        `;
        
        if (this.isReviewMode) {
            header.innerHTML = `
                <div>
                    <h2 style="margin: 0; color: ${this.colors.warning}; font-size: ${this.typography.mainHeaderFontSize}; font-weight: 600; letter-spacing: ${this.typography.mainHeaderLetterSpacing}; text-transform: uppercase;">📋 PLAN REVIEW MODE</h2>
                    <p style="margin: 8px 0 0 0; color: ${this.colors.textLight}; font-size: ${this.typography.subtitleFontSize}; font-weight: 400;">Review orchestrator-generated plan • Modify if needed • Approve or reject</p>
                </div>
                <button id="plan-close-btn" style="background: ${this.colors.danger}; color: white; border: 1px solid ${this.colors.danger}; padding: 12px 20px; border-radius: ${this.borderRadius.button}; cursor: pointer; font-weight: 500; font-size: ${this.typography.buttonFontSize}; text-transform: uppercase; letter-spacing: ${this.typography.buttonLetterSpacing}; transition: ${this.transitions.default}; min-width: 80px;">Close</button>
            `;
        } else {
            header.innerHTML = `
                <div>
                    <h2 style="margin: 0; color: ${this.colors.text}; font-size: ${this.typography.mainHeaderFontSize}; font-weight: 600; letter-spacing: ${this.typography.mainHeaderLetterSpacing}; text-transform: uppercase;">📋 Orchestrator-Generated Execution Plan</h2>
                    <p style="margin: 8px 0 0 0; color: ${this.colors.textLight}; font-size: ${this.typography.subtitleFontSize}; font-weight: 400;">Complex Performance Investigation • ${this.currentPlan.length} Steps • Auto-Generated Dependencies</p>
                </div>
                <div style="background: ${this.colors.accent}; color: white; padding: 8px 12px; border-radius: ${this.borderRadius.badge}; font-size: ${this.typography.badgeFontSize}; font-weight: 500; text-transform: uppercase; letter-spacing: ${this.typography.badgeLetterSpacing};">Interactive Preview</div>
            `;
        }
        
        return header;
    }
    
    /**
     * Create control panel (buttons for actions)
     */
    createControlPanel() {
        const controlPanel = document.createElement('div');
        
        if (this.isReviewMode) {
            controlPanel.innerHTML = `
                <div style="background: ${this.colors.warning}; color: white; padding: 16px; border-radius: 6px; margin-bottom: 20px;">
                    <div style="font-weight: 600; margin-bottom: 8px;">Original Task:</div>
                    <div style="font-size: 14px; opacity: 0.9;">${this.reviewModeData.originalTask}</div>
                    <div style="font-size: 12px; margin-top: 8px; opacity: 0.7;">Created: ${this.reviewModeData.createdAt} • Context: ${this.reviewModeData.contextKey}</div>
                </div>
                
                <div style="display: flex; gap: 16px; margin-bottom: 24px; padding: 20px; background: ${this.colors.panelBackground}; border: 1px solid ${this.colors.borderLight}; border-radius: 3px; align-items: center; flex-wrap: nowrap;">
                    <button id="save-as-is-btn" style="background: ${this.colors.success}; color: white; border: 1px solid ${this.colors.success}; padding: 10px 16px; border-radius: 3px; cursor: pointer; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.2s; min-width: 120px;">💾 Approve Plan</button>
                    <button id="save-modified-btn" style="background: ${this.colors.primary}; color: white; border: 1px solid ${this.colors.primary}; padding: 10px 16px; border-radius: 3px; cursor: pointer; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.2s; min-width: 140px;">✏️ Save Modifications</button>
                    <button id="reject-plan-btn" style="background: ${this.colors.danger}; color: white; border: 1px solid ${this.colors.danger}; padding: 10px 16px; border-radius: 3px; cursor: pointer; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.2s; min-width: 100px;">❌ Reject Plan</button>
                    <div style="margin-left: auto; color: ${this.colors.textLight}; font-size: 13px; text-align: right;">
                        Human approval required for<br><strong>multi-step execution</strong>
                    </div>
                </div>
            `;
        } else {
            // Documentation mode - just explanation
            controlPanel.innerHTML = `
                <div style="background: ${this.colors.accent}; color: white; padding: 16px; border-radius: ${this.borderRadius.panel}; margin-bottom: 20px; font-size: 13px; line-height: 1.5;">
                    <div style="font-weight: 600; margin-bottom: 8px;">🎯 Interactive Demonstration</div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px;">
                        <div><strong>• Dependency Management:</strong> Steps 3-5 use outputs from previous steps</div>
                        <div><strong>• Context Flow:</strong> Data flows through TIME_RANGE → TURBINE_STATUS → WEATHER_DATA → ANALYSIS_RESULTS</div>
                        <div><strong>• Parameter Configuration:</strong> Step 5 includes safety and optimization parameters</div>
                        <div><strong>• Human Approval:</strong> Multi-step plans require explicit approval before execution</div>
                    </div>
                </div>
            `;
        }
        
        return controlPanel;
    }
    
    /**
     * Render execution steps
     */
    renderSteps() {
        const stepsContainer = document.createElement('div');
        stepsContainer.id = 'steps-container';
        stepsContainer.style.cssText = `
            border: 1px solid ${this.colors.border};
            border-radius: ${this.borderRadius.panel};
            min-height: 300px;
            padding: 20px;
            background: ${this.colors.background};
        `;
        
        if (this.currentPlan.length === 0) {
            stepsContainer.innerHTML = `
                <div style="text-align: center; padding: 40px; color: ${this.colors.textLight}; font-style: italic;">
                    No steps configured in this execution plan.
                </div>
            `;
            return stepsContainer;
        }
        
        this.currentPlan.forEach((step, index) => {
            const stepCard = this.createStepCard(step, index);
            stepsContainer.appendChild(stepCard);
        });
        
        return stepsContainer;
    }
    
    /**
     * Create individual step card
     */
    createStepCard(step, index) {
        const stepCard = document.createElement('div');
        stepCard.style.cssText = `
            border: 1px solid ${this.colors.borderLight};
            border-radius: ${this.borderRadius.panel};
            padding: ${this.layout.stepCardPadding};
            margin-bottom: ${this.layout.stepCardBottomMargin};
            background: ${this.colors.panelBackground};
            position: relative;
            box-shadow: ${this.shadows.card};
            transition: ${this.transitions.default};
            cursor: pointer;
        `;
        
        // Add hover effect
        stepCard.addEventListener('mouseenter', () => {
            stepCard.style.borderColor = this.colors.primary;
            stepCard.style.boxShadow = this.shadows.cardHover;
        });
        
        stepCard.addEventListener('mouseleave', () => {
            stepCard.style.borderColor = this.colors.borderLight;
            stepCard.style.boxShadow = this.shadows.card;
        });
        
        // Create inputs display
        let inputsHtml = '';
        if (step.inputs && step.inputs.length > 0) {
            inputsHtml = step.inputs.map(input => {
                const key = Object.keys(input)[0];
                const value = input[key];
                return `<span style="background: ${this.colors.accent}; color: white; padding: 4px 8px; border-radius: ${this.borderRadius.badge}; font-size: ${this.typography.badgeFontSize}; margin-right: 6px; margin-bottom: 4px; display: inline-block; font-weight: 500; text-transform: uppercase; letter-spacing: ${this.typography.badgeLetterSpacing}; border: 1px solid ${this.colors.accent};">${key}: ${value}</span>`;
            }).join('');
        } else {
            inputsHtml = `<span style="color: ${this.colors.textLight}; font-style: italic; font-size: ${this.typography.noInputsFontSize}; text-transform: uppercase; letter-spacing: ${this.typography.noInputsLetterSpacing};">No inputs required</span>`;
        }
        
        // Create parameters display
        let parametersHtml = '';
        if (step.parameters && Object.keys(step.parameters).length > 0) {
            parametersHtml = Object.entries(step.parameters).map(([key, value]) => {
                return `<span style="background: ${this.colors.warning}; color: white; padding: 4px 8px; border-radius: ${this.borderRadius.badge}; font-size: ${this.typography.badgeFontSize}; margin-right: 6px; margin-bottom: 4px; display: inline-block; font-weight: 500; text-transform: uppercase; letter-spacing: ${this.typography.badgeLetterSpacing};">${key}: ${value}</span>`;
            }).join('');
        } else {
            parametersHtml = `<span style="color: ${this.colors.textLight}; font-style: italic; font-size: ${this.typography.noInputsFontSize}; text-transform: uppercase; letter-spacing: ${this.typography.noInputsLetterSpacing};">Default parameters</span>`;
        }
        
        // Create action buttons container
        const actionsContainer = document.createElement('div');
        
        if (this.isReviewMode) {
            // In review mode, add edit and delete buttons
            const editBtn = document.createElement('button');
            editBtn.style.cssText = `
                background: ${this.colors.accent}; 
                color: white; 
                border: 1px solid ${this.colors.accent}; 
                padding: 6px 12px; 
                border-radius: 3px; 
                cursor: pointer; 
                font-size: 11px; 
                font-weight: 500; 
                text-transform: uppercase; 
                letter-spacing: 0.5px; 
                margin-right: 8px; 
                transition: all 0.2s;
            `;
            editBtn.textContent = 'Edit';
            editBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.editStep(index);
            });
            
            const deleteBtn = document.createElement('button');
            deleteBtn.style.cssText = `
                background: ${this.colors.danger}; 
                color: white; 
                border: 1px solid ${this.colors.danger}; 
                padding: 6px 12px; 
                border-radius: 3px; 
                cursor: pointer; 
                font-size: 11px; 
                font-weight: 500; 
                text-transform: uppercase; 
                letter-spacing: 0.5px; 
                transition: all 0.2s;
            `;
            deleteBtn.textContent = 'Delete';
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteStep(index);
            });
            
            actionsContainer.appendChild(editBtn);
            actionsContainer.appendChild(deleteBtn);
        } else {
            // In documentation mode, show expected output badge
            actionsContainer.innerHTML = `<div style="background: ${this.colors.success}; color: white; padding: 4px 8px; border-radius: ${this.borderRadius.badge}; font-size: ${this.typography.badgeFontSize}; font-weight: 500; text-transform: uppercase; letter-spacing: ${this.typography.badgeLetterSpacing};">→ ${step.expected_output}</div>`;
        }
        
        // Create the step card content
        const cardHeader = document.createElement('div');
        cardHeader.style.cssText = `
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: ${this.layout.stepCardHeaderBottomMargin}; 
            padding-bottom: ${this.layout.stepCardHeaderBottomPadding}; 
            border-bottom: 1px solid ${this.colors.borderLight};
        `;
        
        cardHeader.innerHTML = `
            <div style="flex: 1; display: flex; align-items: center;">
                <span style="background: ${this.colors.primary}; color: white; padding: 6px 12px; border-radius: ${this.borderRadius.badge}; font-size: ${this.typography.stepTitleFontSize}; font-weight: 600; text-transform: uppercase; letter-spacing: ${this.typography.stepTitleLetterSpacing};">STEP ${index + 1}</span>
                <span style="margin-left: 12px; font-weight: 600; color: ${this.colors.text}; font-size: ${this.typography.stepCapabilityFontSize}; text-transform: uppercase; letter-spacing: ${this.typography.stepCapabilityLetterSpacing};">${step.capability}</span>
                <span style="margin-left: 12px; color: ${this.colors.textLight}; font-size: 12px; font-weight: normal; border: 1px solid ${this.colors.borderLight}; padding: 2px 6px; border-radius: ${this.borderRadius.badge}; cursor: help;" title="Context Key: Unique identifier for this step's output data">${step.context_key}</span>
            </div>
        `;
        cardHeader.appendChild(actionsContainer);
        
        const cardContent = document.createElement('div');
        cardContent.style.cssText = `display: grid; grid-template-columns: 1fr; gap: ${this.layout.stepCardContentGap};`;
        cardContent.innerHTML = `
            <div>
                <label style="display: flex; align-items: center; font-weight: 600; margin-bottom: 6px; color: ${this.colors.text}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                    Task Objective:
                </label>
                <div style="font-size: 13px; color: ${this.colors.text}; line-height: 1.5; padding: 8px; background: ${this.colors.background}; border: 1px solid ${this.colors.borderLight}; border-radius: ${this.borderRadius.field};">${step.task_objective}</div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 8px;">
                <div>
                    <label style="display: flex; align-items: center; font-weight: 600; margin-bottom: 6px; color: ${this.colors.text}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                        Inputs:
                    </label>
                    <div style="font-size: 13px; line-height: 1.5; padding: 8px; background: ${this.colors.background}; border: 1px solid ${this.colors.borderLight}; border-radius: ${this.borderRadius.field}; min-height: 36px;">${inputsHtml}</div>
                </div>
                <div>
                    <label style="display: flex; align-items: center; font-weight: 600; margin-bottom: 6px; color: ${this.colors.text}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                        Parameters:
                    </label>
                    <div style="font-size: 13px; line-height: 1.5; padding: 8px; background: ${this.colors.background}; border: 1px solid ${this.colors.borderLight}; border-radius: ${this.borderRadius.field}; min-height: 36px;">${parametersHtml}</div>
                </div>
            </div>
        `;
        
        stepCard.appendChild(cardHeader);
        stepCard.appendChild(cardContent);
        
        return stepCard;
    }
    
    /**
     * Edit step functionality (placeholder for review mode)
     */
    editStep(index) {
        // For documentation demo, just show a message
        this.showSuccessNotification(`Edit functionality would open a modal to modify Step ${index + 1}. This is a demo of the approval interface.`);
    }
    
    /**
     * Delete step functionality (placeholder for review mode)
     */
    deleteStep(index) {
        // For documentation demo, just show a message
        this.showSuccessNotification(`Delete functionality would remove Step ${index + 1} from the plan. This is a demo of the approval interface.`);
    }
    
    /**
     * Setup event handlers for review mode
     */
    setupEventHandlers(container) {
        if (this.isReviewMode) {
            const saveAsIsBtn = container.querySelector('#save-as-is-btn');
            const saveModifiedBtn = container.querySelector('#save-modified-btn');
            const rejectBtn = container.querySelector('#reject-plan-btn');
            
            if (saveAsIsBtn) {
                saveAsIsBtn.addEventListener('click', () => {
                    this.showSuccessNotification('✅ Plan approved!\n\nExecuting original plan as proposed by the orchestrator.');
                    if (this.callbacks.onSave) {
                        this.callbacks.onSave({ action: 'approve', plan: this.currentPlan });
                    }
                });
            }
            
            if (saveModifiedBtn) {
                saveModifiedBtn.addEventListener('click', () => {
                    this.showSuccessNotification('✅ Modified plan saved!\n\nExecuting plan with your modifications.');
                    if (this.callbacks.onSave) {
                        this.callbacks.onSave({ action: 'approve_modified', plan: this.currentPlan });
                    }
                });
            }
            
            if (rejectBtn) {
                rejectBtn.addEventListener('click', () => {
                    this.showSuccessNotification('❌ Plan rejected!\n\nExecution cancelled. You can request a different approach.');
                    if (this.callbacks.onCancel) {
                        this.callbacks.onCancel({ action: 'reject' });
                    }
                });
            }
        }
        
        const closeBtn = container.querySelector('#plan-close-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                if (this.callbacks.onCancel) {
                    this.callbacks.onCancel({ action: 'close' });
                }
            });
        }
    }
    
    /**
     * Show success notification
     */
    showSuccessNotification(message) {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${this.colors.success};
            color: white;
            padding: 16px 24px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 10002;
            font-size: 14px;
            font-weight: 500;
            max-width: 400px;
            word-wrap: break-word;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease;
            white-space: pre-line;
        `;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.opacity = '1';
            notification.style.transform = 'translateX(0)';
        }, 10);
        
        // Auto-remove after 4 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 4000);
    }
    
    /**
     * Main render method
     */
    render(parentElement, options = {}) {
        const { container, overlay } = this.createContainer(parentElement, options);
        
        // Create sections
        const header = this.createHeader();
        const controlPanel = this.createControlPanel();
        const stepsContainer = this.renderSteps();
        
        // Assemble the interface
        container.appendChild(header);
        container.appendChild(controlPanel);
        
        // Add steps section header
        const stepsHeader = document.createElement('h3');
        stepsHeader.style.cssText = `
            margin: 0 0 16px 0;
            color: ${this.colors.text};
            font-size: 16px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid ${this.colors.borderLight};
            padding-bottom: 8px;
        `;
        stepsHeader.textContent = 'Execution Steps';
        container.appendChild(stepsHeader);
        container.appendChild(stepsContainer);
        
        // Setup event handlers
        this.setupEventHandlers(container);
        
        return { container, overlay };
    }
}

// Export for use in different contexts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ExecutionPlanViewer;
} else if (typeof window !== 'undefined') {
    window.ExecutionPlanViewer = ExecutionPlanViewer;
} 