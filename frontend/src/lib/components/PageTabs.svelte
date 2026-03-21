<script>
  import { createEventDispatcher } from 'svelte';

  export let items = [];
  export let value = '';
  export let ariaLabel = '选项卡';

  const dispatch = createEventDispatcher();

  function select(id) {
    dispatch('change', { value: id });
  }
</script>

<div class="page-tabs" role="tablist" aria-label={ariaLabel}>
  {#each items as item}
    <button
      type="button"
      role="tab"
      class:active={value === item.id}
      aria-selected={value === item.id}
      on:click={() => select(item.id)}
    >
      {item.title}
    </button>
  {/each}
</div>

<style>
  .page-tabs {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 6px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(15, 23, 42, 0.08);
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
  }

  button {
    border: none;
    background: transparent;
    color: #6f8086;
    padding: 0.8rem 1rem;
    border-radius: 999px;
    transition: background 0.18s ease, color 0.18s ease, transform 0.18s ease;
  }

  button.active {
    color: #17303b;
    background: #fff;
    box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
  }

  button:hover {
    transform: translateY(-1px);
  }

  @media (max-width: 720px) {
    .page-tabs {
      width: 100%;
    }

    button {
      flex: 1 1 0;
      min-width: 88px;
    }
  }
</style>
