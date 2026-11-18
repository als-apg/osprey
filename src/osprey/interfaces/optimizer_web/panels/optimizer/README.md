# OPTIMIZER panel

This folder holds the OPTIMIZER panel. The panel is plain ES modules. It has no build step. The sidecar (`../../app.py`) serves it.

## Layout

The panel has a toolbar and these areas:

- The toolbar names the control target and the environment of the screen. It holds the status chip, the history toggle and one primary action slot.
- The runs list (left, 260 px) shows the job of now, the other open jobs and the stored runs.
- The stage (middle) shows the screen.
- The inspector (right, 340 px) shows the details and the decision of the screen.

Up to 1040 px, the runs list is an overlay and the inspector is a drawer. An edge tab opens the drawer. Escape closes it.

## Module map

| Module | Content |
|---|---|
| `index.html`, `manifest.json`, `css/` | The page, the panel manifest and the style sheets. The sidecar serves them as static files. |
| `panel.js` | Start-up and wiring. It builds the store, the reads, the actions, the shell, the runs list and the view host. It holds the view table `VIEWS`. It draws nothing. |
| `store.js` | The one state object, `reduce(state, event)`, `deriveScreen(state)`, `presentScreen(state)`, `currentScreen(state)`, `runJob(state)`, `createStore()`. |
| `run-model.js` | Pure run figures: evaluations done and planned, the best feasible evaluation so far, the value before the run, the change, the time left. |
| `sync.js` | The reads: the poll tick, an active job beside a newer draft, the runs list, the machine connection, the catalogue (environments and model files). Reads become store events. |
| `actions.js` | What the operator can do, and the reads a view asks for. Every write goes to `relay/...`. |
| `words.js` | The words for what the service names: methods, end actions, what a stop leaves, when a run may write, how a job ended, the status of each screen, directions, a change, and an instant on the facility clock. See [Shared words](#shared-words). |
| `shell.js` | The toolbar, the status chip and its popover, the zones, the drawer. Pure helpers: `chipFor`, `chipDetails`, `inspectorFrame`, `TARGETS`. |
| `runs-list.js` | The runs list. Pure helpers: `runsModel`, `pastRuns` (the stored runs, newest first, filtered; the history table uses it too), `runTime` (the start of a stored run, read as written), `dayLabel` (Today and Yesterday on the facility calendar), `objectiveLabel`, `environmentLabel`. |
| `view-host.js` | The view contract (types) and the host that mounts one view at a time. |
| `knob-bar.js` | The knob bar and the knob name. Use them in every view that shows a knob. |
| `facts-strip.js` | The facts strip (a band of caption and value pairs). |
| `empty-view.js` | The `empty` screen. |
| `setup-view.js`, `setup-form.js`, `setup-notices.js`, `setup-model.js`, `knob-picker.js` | The set-up screens (`setup`, `setup-issues`): the view, the fields of its inspector, its notices (what holds Save and why), its pure helpers around `draft-model.js`, the knob picker. |
| `review-view.js`, `draft-stage.js`, `config-sections.js` | The review screens; the stage of a draft (set-up and review); the settings of a stored config, read only, by section. |
| `run-page.js`, `run-header.js`, `run-knobs.js`, `run-page-model.js` | The run page (`waiting-approval`, `running`, `paused`): the mount, the header with the controls and the cancel confirmation, the inspector with one card per knob and the run details, the pure sentences and facts of the page. |
| `objective-chart.js`, `evaluations-table.js` | The objective chart (each evaluation, best so far, the value before the run) and the evaluations as a table. |
| `outcome-view.js`, `outcome-inspector.js`, `outcome-knobs.js`, `outcome-model.js` | An ended run and a stored run (`completed`, `failed`, `cancelled`, `result`): the view, its inspector, its knobs, and how it ended as pure data. |
| `history-view.js` | The table of stored runs (`history`). |
| `api-client.js`, `poller.js` | The sidecar client and the poll cadence. |
| `approval-model.js`, `approval-env.js`, `draft-model.js` | The rules of the draft, the review and Start; the review's own read of the draft's environment; the form of a draft as pure data. Do not restate their rules. |
| `hyperscript.js` | `h()`, `replaceChildren()`, `fmt()`. Build every node with `h()`. Do not use `innerHTML`. |

## The screens

`deriveScreen(state).id` is one of the 14 ids in `SCREENS`:

| Id | When |
|---|---|
| `empty` | No current job, no set-up. `screen.loading` is true until the first read answers. |
| `setup` | A set-up is open. |
| `setup-issues` | A set-up is open and its last save was refused (`state.setup.issues`). |
| `review` | The current job is a draft (`created`). |
| `review-changed` | The draft changed after it was shown (`approval.needsReview`). This comes first. |
| `review-no-owner` | The session names no account (`ownerKnown === false`) and the draft is not readonly. Start stays enabled. |
| `waiting-approval` | The active job has status `submitted`; or status `running` with `write_authorization.state` `requesting` or `awaiting`; or status `running` with no write authorization and no evaluation yet, unless the draft is readonly (the run has just started). Nothing has been written. |
| `running` | Status `running` (approval granted or not used). Status `cancelling` is also `running`, with `screen.cancelling` true and no control enabled. |
| `paused` | Status `paused`. |
| `completed`, `failed`, `cancelled` | Status `completed`, `error`, `cancelled` of the current job (`ENDINGS` in `words.js`). |
| `history` | The table of stored runs is selected. |
| `result` | A stored run is selected, or a job that is neither active nor the current one (an ended job, another draft). Never an active job. |

Rules:

- While a job is active (`submitted`, `running`, `paused`, `cancelling`), "now" is that run: `waiting-approval`, `running` or `paused`. This is also true when the current job of the service points at a newer draft (the service points it at every new draft). `runJob(state)` gives that job.
- The newer draft shows in the runs list and opens as its review when it is picked. Its Start follows the same gate. The service refuses a second run, and the review shows the reason.
- A job picked in the runs list shows its own screen. An active job gives its run page, with Pause, Resume and Cancel bound to its own id. The current job gives its review or its outcome.
- With no active job, "now" follows the current job. A draft from the agent, or a run that started elsewhere, shows on a panel that shows nothing else.
- A panel that shows a set-up or a past run keeps it. The runs list shows the new job. With a set-up open, a click on a job row shows that job (an active job with its controls). The set-up keeps its edits.
- A set-up of a draft that has started is closed.
- A changed draft needs a new acknowledgement before Start is enabled. This is true for a change by the agent and for a change by the operator. `approval-model.js` holds this rule.

## Store shape

`State` (see the JSDoc in `store.js`):

| Field | Content |
|---|---|
| `service` | `{ ok, error }`. `ok` is null until the first read. |
| `connection` | Whether the environment of the screen is connected on its target. |
| `job` | The current job (`GET jobs/{id}`; `detail=full` once it has left `created`). |
| `activeJob`, `activeConfig` | An active job that is not the current job (`detail=full`), and its stored config. A job that has ended since keeps its last status. Null when the current job is the run or nothing runs. |
| `arrivals` | When the panel saw each evaluation of the run arrive while it was running (for the time left). A read while the run is not running starts the record again. |
| `approval`, `approvalEnv` | The approval state and the draft's own environment read (`approval-model.js`). |
| `owner`, `ownerKnown` | What `relay/session` named. |
| `starting`, `startError` | A Start in flight, and the refusal of the last Start. |
| `control` | `{ pending, error, cancelArmed }` for Pause, Resume and Cancel. |
| `environments`, `models`, `openJobs`, `results` | Lists: `{ list, error }`. `list` is null until the first answer. A failed read keeps the last list and names the error. |
| `envs` | Environment details and presets by name: `{ details, presets, loading, error }`. |
| `readings` | Readings by `readingKey(kind, environment, target)`: `{ values, error, at }`. |
| `past` | Stored runs by run id: `{ details, job, loading, error }`. |
| `selection` | `{ kind: 'present' }`, `{ kind: 'history' }`, `{ kind: 'run', runId }` or `{ kind: 'job', jobId }`. |
| `setup` | `{ id, jobId, environment, target, issues, saving, from, revision }` while a set-up is open. `id` tells one set-up from the next, so a late answer to a save of an earlier set-up does not land on it. `jobId` is null for a new run. `from` is the config the form starts from: the stored draft, or the stored config of a run to run again (`runAgain`); null for a new run. `revision` is the revision of the draft when the set-up opened, so a change by someone else is noticed. |
| `saveNote` | `{ jobId, tone, text, moved }`: the message of the last save, and the knobs whose search window the service moved on save (`movedKnobs` of `draft-model.js`). |
| `filter` | The runs list filter. |

`Screen` (from `deriveScreen`) has: `id`, `loading`, `job`, `config` (the stored config of that job, when read), `subject` (`{ environment, target }` for the toolbar), `gate` and `shown` (review screens), `controls`, `cancelling`, `authorization` (run screens), `figures` (run, outcome and result screens) and `past` (result screen).

## Actions

`actions.js` returns these functions. A view calls them through `ctx.actions`.

| Action | What it does |
|---|---|
| `select(selection)` | Shows the present, the history, a stored run or a job. A stored run is loaded when it is selected. |
| `setFilter(text)` | Filters the stored runs. |
| `newRun()` | Opens an empty set-up. It names no target. |
| `runAgain(config)` | Opens a new set-up from a stored config: its environment and target, with `setup.from` set to the `OptimizationConfig` fields of the config. Writes nothing. A null config does nothing. |
| `editDraft()` | Opens the set-up of the current draft. |
| `closeSetup()` | Closes the set-up and shows the present. |
| `setSetupWhere(environment, target)` | Names the environment and target of the set-up (the toolbar follows). |
| `refuseSetup(issues)` | Holds the set-up with the reasons it cannot be saved. |
| `saveDraft(config, kind)` | Creates (`create`) or updates (`update`) a draft through the relay. One save at a time. On success the set-up closes and the review shows. A refusal holds the set-up with one issue for each reason that the service gives. |
| `markReviewed()` | The operator has looked at the changed draft again. |
| `start(shown)` | Starts the job and revision in `shown` (from `screen.shown` when the view drew). It sends nothing when the store has moved on or the gate holds Start. |
| `pause(jobId?)`, `resume(jobId?)` | Sends the control to the active run (else the current job) when its status allows it. Pass the id of the job the view shows: a control for another job sends nothing. |
| `requestCancel()`, `confirmCancel(jobId?)`, `dismissCancel()` | Cancel has two steps. `confirmCancel` sends only an armed cancel, to the same job as `pause`. |
| `loadCatalog()` | Reads the environments and the model files again, one read at a time. A set-up that opens after a failed read reads them again. |
| `loadEnvironment(name)` | Reads the details and presets into `state.envs`. An entry that was read is kept; an entry that failed is read again. |
| `read(kind, environment, target, names)` | Reads `variables` or `observables` into `state.readings`. |
| `initialBounds(environment, target, preset)` | Returns the initial search windows of the service (not stored). |
| `loadResult(runId)` | Reads a stored run, and its job while the service still holds it, into `state.past`. It reads again while that job has not ended. |
| `refresh()` | Runs the poll at once. |

## Shared words

`words.js` holds the words for what the service names. A view takes these words from it, and does not write its own:

| Export | Content |
|---|---|
| `methodLabel(algorithm)` | The name of a method. No method gives the service default, random search. |
| `endWords(onComplete)` | `{ short, sentence }` of an end action. No end action gives `apply_best`. |
| `STOP_WORDS` | What a cancel or a failure leaves on the machine: `ahead` (before a run), `now` (while a cancel is asked or under way), `after` (a run that stopped), `short` (a settings row). |
| `BEFORE_A_WRITE` | When a run may write. The empty screen and the stage of a draft say it. |
| `ENDINGS`, `endingOf(status)`, `endedWord`, `endedChip`, `endedDot` | How a job ended (`completed`, `error`, `cancelled`): its outcome, its word and its one tone, and that tone on a chip and on a status dot. |
| `SCREEN_STATUS` | For each screen of a job: the chip tone, the label of the toolbar chip and the short word of the runs list. |
| `directionWords(direction)`, `directionWord(direction)` | A direction: its name, the value it looks for ("highest") and which value is better. |
| `signed(value)`, `percent(pct)`, `changeText(figures)` | A signed number, a signed percentage, and the change of the best against the value before the run. |
| `instantText(value, now)` | An instant (`created_at`, `expires_at_ns`) on the facility clock, through `/design-system/js/facility-time.js`: the time of day when the instant is on the facility's today, else the date and the time of day. |

Add a word here when two views say the same thing. Test what the words do (which value has which tone, which count is in a label), not the exact text.

## View contract

A view is one module. It exports a mount function (`MountView` in `view-host.js`):

```js
/** @type {import('./view-host.js').MountView} */
export function mountRunPage(ctx) {
  // Build the DOM once, in ctx.stage and ctx.inspector.
  return {
    update(state, screen) {
      // Called after every store change (about once a second during a run).
      // Redraw only what changed.
    },
    unmount() {
      // Optional: remove listeners and subscriptions.
    },
  };
}
```

`ctx` has:

- `stage`, `inspector`: the two zones. The view owns their children.
- `actions`: see above.
- `shell.setPrimary(spec)`: fills the primary slot of the toolbar. `spec` is `{ label, onClick, disabled, why, detail, id, key }`. The shell rebuilds the button only when `key` changes, so give a `key` that names what the click is bound to (for Start: the job, the revision and the gate). During a run the slot stays empty.
- `shell.fillInspector({ title, desc, tools, body, foot })`: draws the head, body and foot of the inspector. The head has a close button for the narrow layout.
- `shell.setDrawer(open)`, `shell.setDrawerTop(px)`: opens the drawer, or puts its top below a header that must stay in sight.

The host mounts the view that `VIEWS[screen.id]` names. Two screens with the same mount function keep one instance. When the view changes, the host calls `unmount`, empties both zones and empties the primary slot. A view that throws is replaced by a centred error, and the primary slot is emptied. The host mounts the view again on the next update.

Rules for a view:

- Read only `state` and `screen`. Do not call the sidecar. Ask for data with an action. The answer arrives in the store, and the next `update` sees it.
- Compare a key of what you show. Do not rebuild a control under the cursor on each poll.
- Keep Pause and Cancel in the run header, never in the primary slot.
- Leave out an element whose value is null. Do not show a placeholder value.
- Show an instant with `instantText` of `words.js`. Do not format a `Date` yourself: the facility clock applies, not the clock of the viewer. The `timestamp` of a stored run is text on the clock of the service host. Show it as written (`runTime`), and do not convert it.

To add or replace a view, write the module and set its entries in `VIEWS` in `panel.js`. The table has one group of lines per view module.

## Objective chart

`objective-chart.js` draws the objective of a run in plain SVG. The run page and the outcome views use it. The colours come from tokens in `css/objective-chart.css`, so the chart follows the theme.

```js
import { chartInput, chartLegend, mountObjectiveChart } from './objective-chart.js';

const chart = mountObjectiveChart(host); // appends its box to host
chart.update({ ...chartInput(screen.figures, job.data), label, units, xMax: screen.figures.total });
chart.destroy(); // when the view unmounts
```

`update(input)` takes these fields. A field that is null or absent is not drawn.

| Field | Content |
|---|---|
| `records` | The data records, as `job.data` or `data_records` of a stored run give them. |
| `objective`, `direction` | The key of the objective in a record, and `MAXIMIZE` or `MINIMIZE`. An unknown direction gives no best line. |
| `label`, `units` | The name of the objective for the text alternative, and its units. |
| `before` | The value before the run: a dashed line and a label. |
| `best` | `{ index, value }`, a 0-based record index. Absent: the chart finds the best from the records and the direction. `null`: no best point. |
| `bests` | The best so far at each record (the step line). Absent: the chart finds it from the direction alone. Give the figures' `bests`: they follow the constraints, as the service does. |
| `feasible` | One entry per record; `false` draws the point as one that fails a constraint. `null`: the run has no constraint. |
| `end` | `{ index, label, tone }`: a marker after the record at `index` (for example "paused", "cancelled"). `tone` is `success`, `warning`, `error` or `muted`. |
| `xMax` | The fixed x-range, 1 to `xMax` (the planned evaluations). The evaluations not done yet show as a quiet band. Default: the number of records. |
| `firstTries` | The number of first tries: a shaded band. |
| `current` | `live` or `still`: a ring on the last evaluation, which pulses when `live`. |

Rules:

- `chartInput(figures, records)` makes `records`, `objective`, `direction`, `before`, `best`, `bests` and `feasible` from the figures of `run-model.js`. Use it, so the chart names the same best as the facts and the table.
- With no record the chart says that there is no evaluation yet. With records but no finite value of the objective it says that instead.
- Call `update` on each poll. The chart draws again only when its input or its size changes. It moves the points that it drew, so a live run does not flicker.
- The parent sets the size of the box (`.opt-chart`, a flex item). The SVG fills the box and does not change its size.
- The SVG has `role="img"` and a text alternative: the count, the best and the value before the run.
- `chartLegend({ before })` makes the key for a caption. `evaluations-table.js` makes the table of the same records (`evaluationsTable`, `updateEvaluationsTable`).

## CSS

- One file per zone, shared component or view, in `css/`. `index.html` links each file.
- Every colour, radius and spacing is a design-system token, and so is every size that has a token. No raw colour. No pixel font size.
- A size that has no design-system token is a custom property that starts with `--opt-`. Define it once, in the `:root` block at the top of `css/shell.css`, and use it by name. A media or container query cannot read a custom property, so its condition keeps a raw width.
- Use the `osprey-*` primitives of the design system. Do not restyle them.
- The class prefix of this panel is `opt-`.
- Current files: `shell.css`, `runs-list.css`, `knob-bar.css`, `facts-strip.css`, `empty-view.css`, `config-sections.css`, `draft-stage.css`, `setup-view.css`, `review-view.css`, `objective-chart.css`, `run-page.css`, `outcome-view.css`, `history-view.css`.
- Shared helpers in `shell.css`, for every view: `opt-stage-scroll` (a scrolling column in the stage), `opt-stage-fill` (a block that fills the stage), `opt-note` (a plain muted sentence), `opt-flush` (no side padding, for example on an `osprey-section-label`). Do not define them again in a view sheet.

## What the service supplies

A view shows only what the service supplies. This list tells what each element can show.

| Element | Source | Notes |
|---|---|---|
| Knob name | `VariableMetadata.description` | The variable name (the channel identifier) is the second line. No description: the identifier is the only line (`knobLabel`). |
| Safe range, search window, step, per-write limit | `VariableMetadata.min_value`/`max_value`, the stored config's `variable_overrides`, `max_step` | `max_step` is null when the channel has no step limit. |
| Value now (before a run) | `GET environments/{name}/variables` | A failed read gives null. |
| Value before the run | `snapshots.initial` of the job (`detail=full`), `initial_point` of a stored run | Both hold the variables and the objective. Absent: no line and no change. |
| Evaluations done, planned | `data.length`; `total_evaluations`, else `plannedEvaluations(config)` | |
| Best so far, the evaluation that found it | `data`, the objective direction and the stored config's `constraints` | The best feasible evaluation, as the service picks it: every constraint holds (a missing reading fails it) and every knob value is finite. `apply_best` writes that point; a stored run names the point it applied in `best_point`. Unknown direction: no best. |
| Time left | Not supplied | The data records hold no time. `run-model.js` times the evaluations that the panel saw arrive while the run was running; a pause starts the timing again. Fewer than 3: not shown. After a reload the timing starts again. |
| Where a pause takes effect | Not supplied | The status is `paused` at once, but the runner checks for a pause only before the first evaluation, before each guided step and before the final write: not between the first tries. `pauseWords()` of `run-page-model.js` says what a pause does at the point the run is at. |
| A run that has just started | Status `running`, no `write_authorization`, no `data` | Start moves the job to `running` at once. Until the run asks for its write authorization, the panel shows it as waiting (`waiting-approval`): nothing has been written. |
| Step in an evaluation (writing, settling, reading) | Not supplied | Show "Evaluation N of M". |
| Run start time | Stored run: `ResultInfo.timestamp` (local time of the service host) | The panel shows it as written and does not convert it. Present job: `created_at` is the time the draft was made, not the start. It is an instant, and it shows on the facility clock. |
| Who approves, request id, refusal reason | `write_authorization.owner`, `.id`, `.reason`, read by `authorizationNote()` | |
| Writes used | Not supplied | Do not show. |
| Writes asked and allowed, where approved, valid until, channels and ranges | `write_authorization` (`steps`, `approver`, `expires_at_ns`, `channels`, `completion`), read by `authorizationScope()` of `approval-model.js` (`APPROVAL_SURFACES` names the surfaces) | The approval surface and the expiry exist only once the state is `granted`. |
| Time until the request gives up | Not supplied | The `awaiting` record has no deadline. |
| Who started a run, agent or person | Not supplied | The session owner (`relay/session`) is known before Start only. |
| Stored run list (`GET results`) | `run_id`, `environment`, `timestamp`, `objective_name`, `has_model`, `job_id` | No status, no target, no figures, no owner. The service lists a run from its start: the past runs leave out the run of now and a current job that has not ended (`pastRuns`). |
| Stored run (`GET results/{env}/{ts}`) | `config`, `best_point`, `initial_point`, `data_records` | `best_point` and `data_records` can be empty. No status. |
| Status of a stored run | `GET jobs/{job_id}` | The service keeps an ended job for about one hour. After that the status is unknown. |
| Change per stored run in the list | Not supplied | It needs one stored-run read per row. Show it on the selected run only. |
| Open jobs (`GET jobs`) | Jobs that have not ended | Ended jobs are not in this list. Use `GET results` for past runs. |
| Knob not in the limits database | Not supplied | Show the refusal reason after the fact. |
| Restore | Not part of this panel | No button. |

## Checks

```bash
npx vitest run tests/interfaces/optimizer_web --maxWorkers=8
npx eslint src/osprey/interfaces/optimizer_web/panels/optimizer tests/interfaces/optimizer_web
npx prettier --check "src/osprey/interfaces/optimizer_web/panels/optimizer/**/*.{js,css}" "tests/interfaces/optimizer_web/**/*.mjs"
npm run -s typecheck
uv run pytest tests/interfaces/optimizer_web -q
```

Prettier formats the `.js`, `.mjs` and `.css` files only. Do not format `index.html` with Prettier. The Python tests cover the sidecar (`app.py`, the read proxy and the relay).
