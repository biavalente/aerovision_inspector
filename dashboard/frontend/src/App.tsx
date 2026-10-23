import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { api } from "./api";
import type {
  AnnotationRevision,
  AuditEvent,
  BoundingBox,
  Dashboard,
  HumanDecision,
  Inspection,
  TrainingLoop,
} from "./types";

type Tab = "cockpit" | "review" | "history" | "batch";

const stateCopy: Record<string, string> = {
  not_analysed: "Por analisar",
  model_not_ready: "Modelo indisponível",
  likely_normal: "Provavelmente normal",
  likely_anomaly: "Possível anomalia",
  ambiguous: "Caso complexo",
  acceptable: "Captura adequada",
  capture_quality_warning: "Qualidade a confirmar",
  approved_visual: "Aprovada visualmente",
  rejected_visual: "Reprovada visualmente",
  escalated_ndt: "Encaminhada para END",
  recapture_requested: "Nova captura pedida",
  active_baseline: "Baseline ativo",
  active_development: "Desenvolvimento ativo",
  active_type_support: "Diagnóstico de tipo ativo",
  baseline_type_support: "Diagnóstico de tipo anterior",
  validation_failed: "Validação não aprovada",
  validation_required: "Validação necessária",
  queued: "Treino em fila",
  running: "Treino em curso",
  completed: "Concluído",
  ablation: "Ablação / perda de material",
  breakdown: "Degradação superficial",
  fracture: "Fratura",
  groove: "Sulco",
  normal_roi: "Região normal",
  unknown: "Tipo inconclusivo",
  visual_anomaly_region: "Região visual suspeita",
  dent: "Amolgadela (dent)",
  no_roi: "Sem região para classificar",
  not_available: "Classificador de tipo indisponível",
};

const decisionOptions: Array<{ value: HumanDecision; label: string }> = [
  { value: "approved_visual", label: "Aprovar — sem falha visual confirmada" },
  { value: "rejected_visual", label: "Reprovar — falha visual confirmada" },
  { value: "escalated_ndt", label: "Encaminhar para END/análise adicional" },
  { value: "recapture_requested", label: "Pedir nova captura" },
];

const humanBoxLabelOptions = [
  { value: "unknown", label: "Tipo inconclusivo — não usar no treino" },
  { value: "dent", label: "Amolgadela (DLR) — localização, não usar no treino de classes" },
  { value: "ablation", label: "Ablação / perda de material" },
  { value: "breakdown", label: "Degradação superficial" },
  { value: "fracture", label: "Fratura" },
  { value: "groove", label: "Sulco" },
];

function labelFor(value?: string | null) {
  if (!value) return "Por decidir";
  const suggested = value.match(/^tipo sugerido:\s*(.+)$/i);
  const key = suggested?.[1] || value;
  return stateCopy[key] || key.replaceAll("_", " ");
}

function percentage(value?: number | null) {
  return value === undefined || value === null ? "—" : `${Math.round(value * 100)}%`;
}

function time(value?: string | null) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("pt-PT", { dateStyle: "short", timeStyle: "short" }).format(new Date(value));
}

function Badge({ value, kind }: { value?: string | null; kind?: string }) {
  const normalized = (value || "").replaceAll("_", "-");
  return <span className={`badge ${kind || ""} ${normalized}`}>{labelFor(value)}</span>;
}

function Panel({ title, action, children, className = "" }: { title: string; action?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <h2>{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

function Stat({ label, value, hint, alert }: { label: string; value: string | number; hint?: string; alert?: boolean }) {
  return (
    <article className={`stat ${alert ? "stat-alert" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {hint && <small>{hint}</small>}
    </article>
  );
}

function LearningLoop({ title, description, loop }: { title: string; description: string; loop?: TrainingLoop }) {
  const pending = loop?.pending_cases ?? 0;
  const threshold = loop?.threshold ?? 250;
  const progress = Math.min(100, Math.round((pending / Math.max(1, threshold)) * 100));
  return (
    <article className="learning-loop">
      <div><strong>{title}</strong><span>{pending} / {threshold}</span></div>
      <p>{description}</p>
      <div className="learning-track" aria-label={`${title}: ${progress}% do lote`}><i style={{ width: `${progress}%` }} /></div>
      <small>{loop?.eligible_now ? "Lote pronto para criar candidato" : `${loop?.remaining_to_trigger ?? threshold} decisões até ao próximo candidato`}</small>
    </article>
  );
}

type AnnotationBoardProps = {
  inspection: Inspection;
  boxes: BoundingBox[];
  showXai: boolean;
  showBoxes?: boolean;
  editable?: boolean;
  onBoxesChange?: (boxes: BoundingBox[]) => void;
  selectedBox?: number | null;
  onSelectedBox?: (index: number | null) => void;
};

type Point = { x: number; y: number };
type ResizeAnchor = "nw" | "ne" | "se" | "sw";
type BoardInteraction =
  | { kind: "draw"; pointerId: number; start: Point }
  | { kind: "move"; pointerId: number; index: number; start: Point; initial: BoundingBox }
  | { kind: "resize"; pointerId: number; index: number; anchor: ResizeAnchor; initial: BoundingBox };

const clamp = (value: number, minimum: number, maximum: number) => Math.min(maximum, Math.max(minimum, value));

function keepBoxInsideImage(box: BoundingBox): BoundingBox {
  const width = clamp(box.width, 1, 100);
  const height = clamp(box.height, 1, 100);
  return {
    ...box,
    width,
    height,
    x: clamp(box.x, 0, 100 - width),
    y: clamp(box.y, 0, 100 - height),
  };
}

function AnnotationBoard({ inspection, boxes, showXai, showBoxes = true, editable = false, onBoxesChange, selectedBox, onSelectedBox }: AnnotationBoardProps) {
  const board = useRef<HTMLDivElement>(null);
  const [drawing, setDrawing] = useState<BoundingBox | null>(null);
  const drawingRef = useRef<BoundingBox | null>(null);
  const boxesRef = useRef<BoundingBox[]>(boxes);
  const interaction = useRef<BoardInteraction | null>(null);
  const imageUrl = showXai && inspection.xai_url ? inspection.xai_url : inspection.image_url;

  useEffect(() => { boxesRef.current = boxes; }, [boxes]);

  const point = (event: ReactPointerEvent<HTMLElement>): Point => {
    const rect = board.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return {
      x: Math.max(0, Math.min(100, ((event.clientX - rect.left) / rect.width) * 100)),
      y: Math.max(0, Math.min(100, ((event.clientY - rect.top) / rect.height) * 100)),
    };
  };

  const setDrawingBox = (next: BoundingBox | null) => {
    drawingRef.current = next;
    setDrawing(next);
  };

  const commitBoxes = (next: BoundingBox[]) => {
    boxesRef.current = next;
    onBoxesChange?.(next);
  };

  const finishInteraction = () => {
    const active = interaction.current;
    interaction.current = null;
    if (active?.kind !== "draw") {
      setDrawingBox(null);
      return;
    }
    const draft = drawingRef.current;
    if (!draft) return;
    const normalized = {
      ...draft,
      x: Math.min(draft.x, draft.x + draft.width),
      y: Math.min(draft.y, draft.y + draft.height),
      width: Math.abs(draft.width),
      height: Math.abs(draft.height),
    };
    if (normalized.width > 1 && normalized.height > 1 && onBoxesChange) {
      const next = [...boxesRef.current, keepBoxInsideImage({ ...normalized, label: "unknown", source: "human", confidence: null })];
      commitBoxes(next);
      onSelectedBox?.(next.length - 1);
    }
    setDrawingBox(null);
  };

  const startMove = (event: ReactPointerEvent<HTMLDivElement>, index: number) => {
    if (!editable || event.button !== 0) return;
    const initial = boxesRef.current[index];
    if (!initial) return;
    event.preventDefault();
    event.stopPropagation();
    board.current?.setPointerCapture(event.pointerId);
    interaction.current = { kind: "move", pointerId: event.pointerId, index, start: point(event), initial: { ...initial } };
    onSelectedBox?.(index);
  };

  const startResize = (event: ReactPointerEvent<HTMLButtonElement>, index: number, anchor: ResizeAnchor) => {
    if (!editable || event.button !== 0) return;
    const initial = boxesRef.current[index];
    if (!initial) return;
    event.preventDefault();
    event.stopPropagation();
    board.current?.setPointerCapture(event.pointerId);
    interaction.current = { kind: "resize", pointerId: event.pointerId, index, anchor, initial: { ...initial } };
    onSelectedBox?.(index);
  };

  const resizeBox = (initial: BoundingBox, anchor: ResizeAnchor, cursor: Point): BoundingBox => {
    const minimum = 1;
    let left = initial.x;
    let top = initial.y;
    let right = initial.x + initial.width;
    let bottom = initial.y + initial.height;

    if (anchor.includes("w")) left = clamp(cursor.x, 0, right - minimum);
    if (anchor.includes("e")) right = clamp(cursor.x, left + minimum, 100);
    if (anchor.includes("n")) top = clamp(cursor.y, 0, bottom - minimum);
    if (anchor.includes("s")) bottom = clamp(cursor.y, top + minimum, 100);

    return keepBoxInsideImage({ ...initial, x: left, y: top, width: right - left, height: bottom - top });
  };

  return (
    <div
      className={`annotation-board ${editable ? "editable" : ""}`}
      ref={board}
      onPointerDown={(event) => {
        if (!editable || event.button !== 0) return;
        const target = event.target as HTMLElement;
        if (target.closest?.(".region-box, .draw-help")) return;
        const start = point(event);
        event.currentTarget.setPointerCapture(event.pointerId);
        interaction.current = { kind: "draw", pointerId: event.pointerId, start };
        setDrawingBox({ x: start.x, y: start.y, width: 0, height: 0, label: "unknown", source: "human" });
      }}
      onPointerMove={(event) => {
        const active = interaction.current;
        if (!active || active.pointerId !== event.pointerId) return;
        const current = point(event);
        if (active.kind === "draw") {
          const draft = drawingRef.current;
          if (draft) setDrawingBox({ ...draft, width: current.x - active.start.x, height: current.y - active.start.y });
          return;
        }
        const changed = active.kind === "move"
          ? keepBoxInsideImage({
            ...active.initial,
            x: active.initial.x + current.x - active.start.x,
            y: active.initial.y + current.y - active.start.y,
          })
          : resizeBox(active.initial, active.anchor, current);
        commitBoxes(boxesRef.current.map((box, index) => index === active.index ? changed : box));
      }}
      onPointerUp={finishInteraction}
      onPointerCancel={finishInteraction}
    >
      <img src={`${imageUrl}?v=${encodeURIComponent(inspection.updated_at)}`} alt={`Inspeção ${inspection.id}`} draggable={false} />
      {showBoxes && boxes.map((box, index) => (
        <div
          key={`${box.id ?? "draft"}-${index}`}
          className={`region-box ${box.source === "human" ? "human-box" : "model-box"} ${selectedBox === index ? "selected" : ""}`}
          style={{ left: `${box.x}%`, top: `${box.y}%`, width: `${box.width}%`, height: `${box.height}%` }}
          role="button"
          tabIndex={editable ? 0 : -1}
          aria-label={`Região ${index + 1}: ${labelFor(box.label)}`}
          onPointerDown={(event) => startMove(event, index)}
          onClick={(event) => {
            event.stopPropagation();
            onSelectedBox?.(index);
          }}
        >
          <span>{labelFor(box.label)}</span>
          {editable && selectedBox === index && (["nw", "ne", "se", "sw"] as ResizeAnchor[]).map((anchor) => (
            <button
              type="button"
              className={`resize-handle resize-${anchor}`}
              key={anchor}
              aria-label={`Redimensionar caixa pelo canto ${anchor}`}
              onPointerDown={(event) => startResize(event, index, anchor)}
            />
          ))}
        </div>
      ))}
      {drawing && (
        <div
          className="region-box drawing-box"
          style={{
            left: `${Math.min(drawing.x, drawing.x + drawing.width)}%`,
            top: `${Math.min(drawing.y, drawing.y + drawing.height)}%`,
            width: `${Math.abs(drawing.width)}%`,
            height: `${Math.abs(drawing.height)}%`,
          }}
        />
      )}
      {editable && <div className="draw-help">Arrasta numa área vazia para criar · arrasta uma caixa para mover</div>}
    </div>
  );
}

function InspectionList({ items, currentId, selectedIds, onSelect, onToggle }: {
  items: Inspection[];
  currentId?: number;
  selectedIds: Set<number>;
  onSelect: (id: number) => void;
  onToggle: (id: number) => void;
}) {
  return (
    <div className="inspection-list">
      {items.map((inspection) => (
        <div className={`inspection-row ${inspection.id === currentId ? "active" : ""}`} key={inspection.id}>
          <label className="select-check" title="Selecionar para decisão em lote">
            <input type="checkbox" checked={selectedIds.has(inspection.id)} onChange={() => onToggle(inspection.id)} />
          </label>
          <button type="button" className="inspection-main" onClick={() => onSelect(inspection.id)}>
            <span className="inspection-title">{inspection.source_name}</span>
            <span className="inspection-meta">#{inspection.id} · {time(inspection.updated_at)}</span>
          </button>
          <div className="inspection-tags">
            <Badge value={inspection.ai_state} />
            {inspection.final_decision && <Badge value={inspection.final_decision} />}
          </div>
        </div>
      ))}
      {!items.length && <div className="empty-state">Ainda não há imagens. Carrega um ficheiro ou prepara o conjunto de demonstração.</div>}
    </div>
  );
}

function Metric({ label, value, warning }: { label: string; value: string; warning?: boolean }) {
  return <div className={`metric ${warning ? "metric-warning" : ""}`}><span>{label}</span><strong>{value}</strong></div>;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("cockpit");
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [items, setItems] = useState<Inspection[]>([]);
  const [currentId, setCurrentId] = useState<number | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showModelBoxes, setShowModelBoxes] = useState(true);
  const [draftBoxes, setDraftBoxes] = useState<BoundingBox[]>([]);
  const [selectedBox, setSelectedBox] = useState<number | null>(null);
  const [decision, setDecision] = useState<HumanDecision>("approved_visual");
  const [reason, setReason] = useState("");
  const [actor, setActor] = useState("Inspetor de demonstração");
  const [history, setHistory] = useState<{ events: AuditEvent[]; annotation_revisions: AnnotationRevision[] } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ text: string; error?: boolean } | null>(null);
  const uploadInput = useRef<HTMLInputElement>(null);

  const current = useMemo(() => items.find((inspection) => inspection.id === currentId) || null, [items, currentId]);

  const refresh = async (keepCurrent = true) => {
    const [nextDashboard, inspectionData] = await Promise.all([api.dashboard(), api.inspections()]);
    setDashboard(nextDashboard);
    setItems(inspectionData.items);
    if (!keepCurrent || !currentId || !inspectionData.items.some((item) => item.id === currentId)) {
      setCurrentId(inspectionData.items[0]?.id ?? null);
    }
  };

  const run = async (key: string, action: () => Promise<void>) => {
    setBusy(key);
    setNotice(null);
    try {
      await action();
    } catch (error) {
      setNotice({ text: error instanceof Error ? error.message : "Ocorreu um erro inesperado.", error: true });
    } finally {
      setBusy(null);
    }
  };

  useEffect(() => { void run("startup", () => refresh(false)); }, []);

  useEffect(() => {
    if (!current) return;
    const existing = current.human_boxes.length ? current.human_boxes : current.model_boxes.map((box) => ({ ...box, source: "human" as const }));
    const knownLabels = new Set(humanBoxLabelOptions.map((option) => option.value));
    setDraftBoxes(existing.map((box) => ({
      ...box,
      label: knownLabels.has(box.label) ? box.label : "unknown",
    })));
    setSelectedBox(null);
    setDecision(current.final_decision || "approved_visual");
    setReason(current.final_reason || "");
  }, [current?.id, current?.updated_at]);

  useEffect(() => {
    if (tab !== "history" || !current) return;
    void run("history", async () => setHistory(await api.history(current.id)));
  }, [tab, current?.id]);

  const toggleSelected = (id: number) => setSelectedIds((previous) => {
    const next = new Set(previous);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });

  const analyseCurrent = () => {
    if (!current) return;
    void run("analyse", async () => {
      await api.analyse(current.id);
      await refresh();
      setNotice({ text: "Análise local concluída. A decisão final continua pendente de revisão humana." });
    });
  };

  const uploadImages = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    event.target.value = "";
    if (!files.length) return;
    void run("upload", async () => {
      const uploaded: Inspection[] = [];
      const failed: string[] = [];
      for (const file of files) {
        try {
          uploaded.push(await api.upload(file));
        } catch (error) {
          failed.push(`${file.name}: ${error instanceof Error ? error.message : "erro no envio"}`);
        }
      }
      if (!uploaded.length) throw new Error(failed.join(" · ") || "Não foi possível adicionar as imagens.");
      await refresh(false);
      setCurrentId(uploaded[0].id);
      setNotice({
        text: `${uploaded.length} ${uploaded.length === 1 ? "imagem carregada e analisada automaticamente" : "imagens carregadas e analisadas automaticamente"}.${failed.length ? ` ${failed.length} não foram adicionadas.` : ""}`,
        error: false,
      });
    });
  };

  const saveReview = () => {
    if (!current) return;
    void run("review", async () => {
      const result = await api.review(current.id, { decision, reason, actor, boxes: draftBoxes.map((box) => ({ ...box, source: "human" })) });
      await refresh();
      setNotice({
        text: result.training_trigger
          ? `Revisão guardada e candidato de aprendizagem contínua criado para validação (${result.training_trigger.training_target === "primary normality autoencoder only" ? "modelo primário" : "classificador secundário"}).`
          : "Revisão, caixas e decisão final guardadas.",
      });
    });
  };

  const updateDraftLabel = (index: number, label: string) => {
    setDraftBoxes((previous) => previous.map((box, boxIndex) => {
      if (boxIndex !== index) return box;
      return { ...box, label };
    }));
  };

  const removeDraftBox = (index: number) => {
    setDraftBoxes((previous) => previous.filter((_, position) => position !== index));
    setSelectedBox((previous) => previous === index ? null : previous !== null && previous > index ? previous - 1 : previous);
  };

  const submitBatch = (batchDecision: HumanDecision) => {
    if (!selectedIds.size) {
      setNotice({ text: "Seleciona pelo menos uma imagem para a decisão em lote.", error: true });
      return;
    }
    void run("batch", async () => {
      const result = await api.batchDecision({
        inspection_ids: [...selectedIds], decision: batchDecision, reason, actor,
      });
      setSelectedIds(new Set());
      await refresh();
      setNotice({ text: `${result.updated_ids.length} decisões guardadas. As caixas individuais não foram alteradas.` });
    });
  };

  const navItems: Array<{ id: Tab; icon: string; label: string }> = [
    { id: "cockpit", icon: "◈", label: "Inspeção" },
    { id: "review", icon: "⌁", label: "Revisão humana" },
    { id: "history", icon: "◷", label: "Histórico" },
    { id: "batch", icon: "▦", label: "Decisão em lote" },
  ];

  return (
    <main className="app-shell">
      <aside className={`sidebar ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
        <div className="sidebar-top">
          {!sidebarCollapsed && <div className="brand"><span className="brand-mark">A</span><div><strong>AeroVision</strong><small>Inspector</small></div></div>}
          <button className="sidebar-toggle" type="button" onClick={() => setSidebarCollapsed((value) => !value)} aria-label={sidebarCollapsed ? "Mostrar menu lateral" : "Esconder menu lateral"} title={sidebarCollapsed ? "Mostrar menu" : "Esconder menu"}>{sidebarCollapsed ? "›" : "‹"}</button>
        </div>
        <input ref={uploadInput} onChange={uploadImages} type="file" accept="image/png,image/jpeg,image/bmp,image/tiff" multiple hidden />
        {!sidebarCollapsed && <>
          <nav>{navItems.map((item) => <button key={item.id} className={tab === item.id ? "nav-active" : ""} onClick={() => setTab(item.id)}><span>{item.icon}</span>{item.label}</button>)}</nav>
          <div className="sidebar-actions">
            <button className="sidebar-action" onClick={() => uploadInput.current?.click()} disabled={Boolean(busy)}>＋ Adicionar imagens</button>
          </div>
        </>}
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div><p className="eyebrow">CENTRO DE INSPEÇÃO VISUAL</p><h1>{navItems.find((item) => item.id === tab)?.label}</h1></div>
        </header>

        {notice && <div className={`notice ${notice.error ? "notice-error" : ""}`}><span>{notice.error ? "!" : "✓"}</span>{notice.text}<button onClick={() => setNotice(null)}>×</button></div>}

        {tab === "cockpit" && <>
          <div className="stats-grid">
            <Stat label="Na fila" value={dashboard?.counts.total ?? "—"} hint="imagens registadas" />
            <Stat label="Analisadas" value={dashboard?.counts.analysed ?? "—"} hint="pelo modelo local" />
            <Stat label="Revisão humana" value={dashboard?.counts.awaiting_human_decision ?? "—"} hint="a aguardar decisão" />
            <Stat label="Casos complexos" value={dashboard?.counts.ambiguous ?? "—"} hint="requerem contexto adicional" alert={(dashboard?.counts.ambiguous || 0) > 0} />
          </div>
          <div className="cockpit-grid">
            <Panel title="Fila de inspeção" action={<span className="small-muted">{items.length} imagens</span>} className="queue-panel">
              <InspectionList items={items} currentId={currentId ?? undefined} selectedIds={selectedIds} onSelect={setCurrentId} onToggle={toggleSelected} />
            </Panel>
            <Panel title="Evidência da análise" className="analysis-panel">
              {current ? <>
                <div className="inspection-head"><div><strong>{current.source_name}</strong><p>Perfil: pá/tipo metálico · ID #{current.id}</p></div><div className="head-badges"><Badge value={current.ai_state} /><Badge value={current.quality_state} /></div></div>
                <div className="metrics-grid">
                  <Metric label={current.quality.active_model_family === "rtdetr_dlr_detector" ? "Score RT-DETR" : "Índice de anomalia"} value={percentage(current.anomaly_score)} />
                  <Metric label="Confiança operacional" value={percentage(current.operational_confidence)} />
                  <Metric label="Fiabilidade da imagem" value={percentage(current.quality.reliability)} warning={current.quality_state === "capture_quality_warning"} />
                  <Metric label="Brilho especular" value={current.quality.glare_fraction === undefined ? "—" : percentage(current.quality.glare_fraction)} warning={(current.quality.glare_fraction || 0) > 0.12} />
                </div>
                <div className="evidence-comparison">
                  <div className="evidence-frame"><div className="evidence-frame-heading"><strong>Imagem original</strong><span>captura registada</span></div><AnnotationBoard inspection={current} boxes={current.model_boxes} showXai={false} showBoxes={false} /></div>
                  <div className="evidence-frame"><div className="evidence-frame-heading evidence-frame-heading-controls"><div><strong>Mapa de evidência</strong><span>{current.xai_url ? current.quality.active_model_family === "rtdetr_dlr_detector" ? "evidência espacial das caixas RT-DETR" : "evidência do modelo histórico" : "disponível após análise"}</span></div><div className="evidence-toggles"><label title="Mostrar ou ocultar o mapa de calor"><input type="checkbox" checked={showHeatmap} onChange={(event) => setShowHeatmap(event.target.checked)} />Calor</label><label title="Mostrar ou ocultar as caixas sugeridas pelo modelo"><input type="checkbox" checked={showModelBoxes} onChange={(event) => setShowModelBoxes(event.target.checked)} />Caixas</label></div></div><AnnotationBoard inspection={current} boxes={current.model_boxes} showXai={showHeatmap} showBoxes={showModelBoxes} /></div>
                </div>
                <div className="analysis-actions"><button className="button primary" onClick={analyseCurrent} disabled={Boolean(busy)}>{busy === "analyse" ? "A analisar…" : current.ai_state === "not_analysed" ? "Executar análise" : "Analisar novamente"}</button><button className="button secondary" onClick={() => setTab("review")}>Abrir revisão humana</button></div>
                {current.quality.detector_status === "available" && <div className="type-proposal">
                  <div><span>Detetor ativo</span><strong>RT-DETR · amolgadela</strong></div>
                  <div><span>Score / limiar</span><strong>{percentage(current.quality.detector_confidence)} / {percentage(current.quality.detector_threshold)}</strong></div>
                  <p>{current.quality.detector_box_count
                    ? `${current.quality.detector_box_count} caixa(s) mantida(s) acima do limiar. O modelo foi avaliado em DLR Aircraft Dent; a delimitação continua a ser uma hipótese a confirmar pelo inspetor.`
                    : "Nenhuma caixa ultrapassou o limiar operacional. Isto é evidência para priorização, não aprovação automática da peça."}</p>
                </div>}
                {current.quality.detector_status !== "available" && current.quality.predicted_defect_type && <div className="type-proposal">
                  <div><span>Tipo visual sugerido</span><strong>{labelFor(current.quality.predicted_defect_type)}</strong></div>
                  <div><span>Confiança calibrada</span><strong>{percentage(current.quality.predicted_defect_type_confidence)}</strong></div>
                  <p>{current.quality.defect_type_scan?.conflict
                    ? "Foram encontradas hipóteses visuais conflituantes; esta imagem segue como caso complexo para revisão humana."
                    : "Classificação obtida por varredura local em várias escalas. É apoio à revisão humana; não altera a decisão final."}</p>
                </div>}
                <div className="explain-box"><strong>Porque é que o modelo sinalizou isto</strong><p>{current.quality.xai_interpretation || "Executa a análise para obter o mapa de evidência e os indicadores da captura."}</p>{current.quality.decision_basis && <ul>{current.quality.decision_basis.map((basis) => <li key={basis}>{basis}</li>)}</ul>}</div>
              </> : <div className="empty-state large">Prepara a demonstração ou adiciona uma imagem para começar.</div>}
            </Panel>
          </div>
        </>}

        {tab === "review" && <div className="review-layout">
          <Panel title="Imagem e caixas de delimitação" action={current && <Badge value={current.ai_state} />} className="review-image-panel">
            {current ? <><AnnotationBoard inspection={current} boxes={draftBoxes} showXai={false} editable onBoxesChange={setDraftBoxes} selectedBox={selectedBox} onSelectedBox={setSelectedBox} /><p className="annotation-note">Arrasta numa área vazia para criar uma caixa, arrasta uma caixa para a mover e usa os quatro cantos para redimensioná-la. Guardar cria uma nova revisão sem apagar o histórico.</p></> : <div className="empty-state">Seleciona uma inspeção na fila.</div>}
          </Panel>
          <div className="review-side">
            <Panel title="Decisão final"><div className="form-stack"><label>Decisão<select value={decision} onChange={(event) => setDecision(event.target.value as HumanDecision)}>{decisionOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label><label>Justificação<textarea value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Regista a evidência que sustentou a decisão…" rows={4} /></label><label>Revisor<input value={actor} onChange={(event) => setActor(event.target.value)} /></label><button className="button primary full" onClick={saveReview} disabled={!current || Boolean(busy)}>{busy === "review" ? "A guardar…" : "Guardar decisão e revisão"}</button></div></Panel>
            <Panel title={`Caixas humanas (${draftBoxes.length})`} action={<button className="text-button" onClick={() => { setDraftBoxes([]); setSelectedBox(null); }}>Limpar</button>}><div className="box-editor">{draftBoxes.map((box, index) => <div className={`box-list-item ${selectedBox === index ? "box-form-selected" : ""}`} key={`${box.id ?? "draft"}-${index}`}><button type="button" className="box-select" onClick={() => setSelectedBox(index)}><span className="box-swatch" /><span><strong>Região {index + 1}</strong><small>{labelFor(box.label)}</small></span></button><button className="icon-button" type="button" aria-label={`Apagar região ${index + 1}`} onClick={() => removeDraftBox(index)}>×</button></div>)}{selectedBox !== null && draftBoxes[selectedBox] && <div className="box-label-editor"><label>Tipo da região<select value={draftBoxes[selectedBox].label} onChange={(event) => updateDraftLabel(selectedBox, event.target.value)}>{humanBoxLabelOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label><p>Escolhe um tipo confirmado para o tornar elegível para aprendizagem ativa. A posição e as dimensões são editadas diretamente sobre a imagem.</p></div>}{!draftBoxes.length && <div className="empty-state compact">Desenha uma caixa ou arrasta as sugestões do modelo.</div>}</div></Panel>
          </div>
        </div>}

        {tab === "history" && <div className="history-grid">
          <Panel title="Selecionar inspeção"><InspectionList items={items} currentId={currentId ?? undefined} selectedIds={selectedIds} onSelect={setCurrentId} onToggle={toggleSelected} /></Panel>
          <Panel title={current ? `Rasto auditável · #${current.id}` : "Rasto auditável"}>{current ? <div className="timeline">{history?.events.map((event) => <article className="timeline-event" key={event.id}><div className="timeline-dot" /><div><div className="timeline-meta"><strong>{event.event_type.replaceAll("_", " ")}</strong><span>{time(event.created_at)} · {event.actor}</span></div><p>{Object.entries(event.payload).map(([key, value]) => `${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`).join(" · ")}</p><small>Hash: {event.event_hash.slice(0, 16)}…</small></div></article>)}{!history && <div className="empty-state">A carregar histórico…</div>}{history && !history.events.length && <div className="empty-state">Ainda não existem eventos.</div>}</div> : <div className="empty-state">Seleciona uma inspeção.</div>}</Panel>
          <div className="history-side-stack">
            <Panel title="Aprendizagem contínua"><div className="learning-summary"><LearningLoop title="Modelo primário" description="Apenas aprovações humanas sem defeito." loop={dashboard?.training.primary_normal_learning} /><LearningLoop title="Classificador secundário" description="Reprovações ou casos complexos; tipos vêm de caixas humanas." loop={dashboard?.training.secondary_rejected_learning} /><p>Os dois ciclos criam candidatos locais para validação; nenhum substitui o modelo ativo automaticamente.</p></div></Panel>
            <Panel title="Revisões de anotação"><div className="revision-list">{history?.annotation_revisions.map((revision) => <article className="revision" key={revision.id}><strong>Revisão {revision.revision_number}</strong><span>{revision.boxes.length} caixas · {revision.actor} · {time(revision.created_at)}</span>{revision.note && <p>{revision.note}</p>}</article>) || <div className="empty-state compact">As revisões aparecerão aqui.</div>}</div></Panel>
          </div>
        </div>}

        {tab === "batch" && <div className="batch-layout">
          <Panel title="Seleção para decisão em lote" action={<span className="small-muted">{selectedIds.size} selecionadas</span>}><p className="panel-intro">A ação em lote só define a decisão e a justificação comuns. As caixas de delimitação não são copiadas nem substituídas.</p><div className="batch-actions"><select value={decision} onChange={(event) => setDecision(event.target.value as HumanDecision)}>{decisionOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select><input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Justificação comum" /><button className="button primary" onClick={() => submitBatch(decision)} disabled={!selectedIds.size || Boolean(busy)}>{busy === "batch" ? "A guardar…" : "Aplicar a selecionadas"}</button></div><InspectionList items={items} currentId={currentId ?? undefined} selectedIds={selectedIds} onSelect={setCurrentId} onToggle={toggleSelected} /></Panel>
        </div>}

      </section>
    </main>
  );
}
