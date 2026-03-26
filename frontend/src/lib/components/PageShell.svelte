<script>
  export let title = '';
  export let subtitle = '';
</script>

<section class="page-shell">
  <header class="page-shell__header">
    <div class="page-shell__copy">
      <h2>{title}</h2>
      {#if subtitle}
        <p>{subtitle}</p>
      {/if}
    </div>

    {#if $$slots.meta || $$slots.actions}
      <div class="page-shell__aside">
        {#if $$slots.meta}
          <div class="page-shell__meta">
            <slot name="meta" />
          </div>
        {/if}

        {#if $$slots.actions}
          <div class="page-shell__actions">
            <slot name="actions" />
          </div>
        {/if}
      </div>
    {/if}
  </header>

  {#if $$slots.toolbar}
    <div class="page-shell__toolbar">
      <slot name="toolbar" />
    </div>
  {/if}

  <div class="page-shell__body">
    <slot />
  </div>
</section>

<style>
  .page-shell {
    display: grid;
    gap: 12px;
    align-content: start;
  }

  .page-shell__header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px 16px;
    padding: 0 0 12px;
    border-radius: 0;
    background: transparent;
    border: 0;
    border-bottom: 1px solid rgba(15, 23, 42, 0.08);
    box-shadow: none;
  }

  .page-shell__copy {
    display: grid;
    gap: 2px;
    flex: 1 1 240px;
    min-width: 0;
  }

  .page-shell__copy h2 {
    margin: 0;
    font-size: 1rem;
    letter-spacing: -0.02em;
    font-family: 'IBM Plex Sans', sans-serif;
  }

  .page-shell__copy p {
    margin: 0;
    color: #6f8086;
    font-size: 0.88rem;
  }

  .page-shell__aside {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: flex-end;
    gap: 8px 10px;
    margin-left: auto;
    flex: 1 1 320px;
    min-width: 0;
  }

  .page-shell__meta,
  .page-shell__actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }

  .page-shell__toolbar {
    min-width: 0;
  }

  .page-shell__body {
    display: grid;
    gap: 12px;
  }

  @media (max-width: 720px) {
    .page-shell {
      gap: 12px;
    }

    .page-shell__header {
      padding: 0 0 12px;
    }

    .page-shell__aside {
      width: 100%;
      justify-content: flex-start;
      margin-left: 0;
    }
  }
</style>
