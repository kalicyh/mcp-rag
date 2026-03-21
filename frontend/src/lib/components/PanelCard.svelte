<script>
  export let title = '';
  export let subtitle = '';
  export let compact = false;
</script>

<section class="panel-card card">
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
  .panel-card__header {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: flex-start;
    padding: 18px 20px 0;
  }

  .panel-card__body {
    padding: 18px 20px 20px;
  }

  .panel-card__body.compact {
    padding-top: 16px;
  }

  .panel-card__title {
    margin: 0;
    font-size: 1rem;
    font-weight: 700;
  }

  .panel-card__subtitle {
    margin: 0.35rem 0 0;
    color: #6f8086;
  }

  @media (max-width: 720px) {
    .panel-card__header,
    .panel-card__body {
      padding-left: 16px;
      padding-right: 16px;
    }
  }
</style>
