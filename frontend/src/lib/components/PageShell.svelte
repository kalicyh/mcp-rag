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
    gap: 18px;
  }

  .page-shell__header {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    justify-content: space-between;
    gap: 14px 18px;
    padding: 18px 20px;
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.74);
    border: 1px solid rgba(15, 23, 42, 0.06);
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
  }

  .page-shell__copy {
    display: grid;
    gap: 4px;
  }

  .page-shell__copy h2 {
    margin: 0;
    font-size: 1.18rem;
    letter-spacing: -0.02em;
    font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif;
  }

  .page-shell__copy p {
    margin: 0;
    color: #6f8086;
    font-size: 0.92rem;
  }

  .page-shell__aside {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: flex-end;
    gap: 10px 12px;
    margin-left: auto;
  }

  .page-shell__meta,
  .page-shell__actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
  }

  .page-shell__toolbar {
    min-width: 0;
  }

  .page-shell__body {
    display: grid;
    gap: 18px;
  }

  @media (max-width: 720px) {
    .page-shell {
      gap: 16px;
    }

    .page-shell__header {
      padding: 16px;
    }

    .page-shell__aside {
      width: 100%;
      justify-content: flex-start;
      margin-left: 0;
    }
  }
</style>
