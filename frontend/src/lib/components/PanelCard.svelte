<script>
  export let title = '';
  export let subtitle = '';
  export let compact = false;
  export let fill = false;
</script>

<section class:fill class="panel-card card">
  {#if title || subtitle || $$slots.actions}
    <header class="panel-card__header">
      {#if title || subtitle}
        <div>
          {#if title}
            <h3 class="panel-card__title">{title}</h3>
          {/if}
          {#if subtitle}
            <p class="panel-card__subtitle">{subtitle}</p>
          {/if}
        </div>
      {/if}

      {#if $$slots.actions}
        <div class="card-actions">
          <slot name="actions" />
        </div>
      {/if}
    </header>
  {/if}

  <div class:compact class="panel-card__body">
    <slot />
  </div>
</section>

<style>
  .panel-card.fill {
    min-height: 100%;
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
  }

  .panel-card__header {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: flex-start;
    padding: 14px 16px 0;
  }

  .panel-card__body {
    padding: 14px 16px 16px;
  }

  .panel-card.fill .panel-card__body {
    min-height: 0;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .panel-card__body.compact {
    padding-top: 12px;
  }

  .panel-card__title {
    margin: 0;
    font-size: 0.96rem;
    font-weight: 600;
  }

  .panel-card__subtitle {
    margin: 0.2rem 0 0;
    color: #6f8086;
    font-size: 0.84rem;
  }

  @media (max-width: 720px) {
    .panel-card__header,
    .panel-card__body {
      padding-left: 16px;
      padding-right: 16px;
    }
  }
</style>
