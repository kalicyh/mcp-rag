<script>
  import { createEventDispatcher, onMount } from 'svelte';

  export let options = [];
  export let value = '';
  export let placeholder = '请选择';
  export let ariaLabel = '下拉选择';
  export let disabled = false;

  const dispatch = createEventDispatcher();
  let root;
  let open = false;
  let searchQuery = '';
  let searchInput;

  $: normalizedOptions = options.map((option) =>
    typeof option === 'string'
      ? { value: option, label: option }
      : {
          value: option?.value ?? '',
          label: option?.label ?? String(option?.value ?? ''),
        }
  );

  $: selectedOption = normalizedOptions.find((option) => option.value === value);
  $: selectedLabel = selectedOption?.label ?? placeholder;
  $: filteredOptions = normalizedOptions.filter((option) => {
    const query = String(searchQuery || '').trim().toLowerCase();
    if (!query) return true;
    return [option.label, option.value]
      .filter(Boolean)
      .some((item) => String(item).toLowerCase().includes(query));
  });

  function toggle() {
    if (disabled) return;
    open = !open;
    if (open) {
      searchQuery = '';
      queueMicrotask(() => searchInput?.focus());
    }
  }

  function close() {
    open = false;
  }

  function choose(nextValue) {
    value = nextValue;
    open = false;
    searchQuery = '';
    dispatch('change', { value: nextValue });
  }

  function handleTriggerKeydown(event) {
    if (disabled) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggle();
    }
    if (event.key === 'Escape') {
      close();
    }
  }

  function handleDocumentPointer(event) {
    if (!root?.contains(event.target)) {
      close();
    }
  }

  onMount(() => {
    document.addEventListener('pointerdown', handleDocumentPointer);
    return () => document.removeEventListener('pointerdown', handleDocumentPointer);
  });
</script>

<div class="select-field" bind:this={root}>
  <button
    type="button"
    class:open
    class="select-field__trigger"
    aria-expanded={open}
    aria-haspopup="listbox"
    aria-label={ariaLabel}
    disabled={disabled}
    on:click={toggle}
    on:keydown={handleTriggerKeydown}
  >
    <span class:selected={!!selectedOption} class="select-field__value">{selectedLabel}</span>
    <span class="select-field__chevron" aria-hidden="true"></span>
  </button>

  {#if open}
    <div class="select-field__menu" role="listbox" aria-label={ariaLabel}>
      {#if normalizedOptions.length >= 8}
        <div class="select-field__search">
          <input
            bind:this={searchInput}
            bind:value={searchQuery}
            class="select-field__search-input"
            type="text"
            placeholder="输入筛选..."
            on:click|stopPropagation
            on:keydown|stopPropagation
          />
        </div>
      {/if}
      <div class="select-field__options">
        {#each filteredOptions as option}
          <button
            type="button"
            role="option"
            class:selected={option.value === value}
            class="select-field__option"
            aria-selected={option.value === value}
            on:click={() => choose(option.value)}
          >
            <span>{option.label}</span>
            {#if option.value === value}
              <span class="select-field__check" aria-hidden="true">✓</span>
            {/if}
          </button>
        {:else}
          <div class="select-field__empty">没有匹配项</div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .select-field {
    position: relative;
  }

  .select-field__trigger {
    width: 100%;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.9);
    color: var(--text);
    padding: 0.8rem 0.95rem;
    min-height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    box-shadow: none;
    transform: none;
  }

  .select-field__trigger.open,
  .select-field__trigger:hover,
  .select-field__trigger:focus-visible {
    border-color: rgba(15, 118, 110, 0.45);
    box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.1);
    outline: none;
    transform: none;
  }

  .select-field__value {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    text-align: left;
    color: var(--muted);
  }

  .select-field__value.selected {
    color: var(--text);
  }

  .select-field__chevron {
    width: 10px;
    height: 10px;
    flex: 0 0 auto;
    border-right: 2px solid currentColor;
    border-bottom: 2px solid currentColor;
    transform: rotate(45deg) translateY(-2px);
    opacity: 0.75;
    margin-right: 4px;
  }

  .select-field__menu {
    position: absolute;
    z-index: 20;
    top: calc(100% + 8px);
    left: 0;
    right: 0;
    padding: 8px;
    border-radius: 18px;
    border: 1px solid rgba(15, 23, 42, 0.08);
    background: rgba(255, 253, 249, 0.98);
    box-shadow: 0 18px 36px rgba(15, 23, 42, 0.12);
    backdrop-filter: blur(18px);
    display: grid;
    gap: 8px;
  }

  .select-field__search {
    padding: 2px;
  }

  .select-field__search-input {
    width: 100%;
    min-height: 42px;
    border: 1px solid rgba(15, 23, 42, 0.1);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.94);
    color: var(--text);
    padding: 0.68rem 0.8rem;
    outline: none;
    box-shadow: none;
  }

  .select-field__search-input:focus {
    border-color: rgba(15, 118, 110, 0.45);
    box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.1);
  }

  .select-field__options {
    max-height: min(420px, 48vh);
    overflow-y: auto;
    padding-right: 2px;
  }

  .select-field__option {
    width: 100%;
    border: none;
    background: transparent;
    color: var(--text);
    border-radius: 12px;
    padding: 0.82rem 0.9rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    text-align: left;
    box-shadow: none;
    transform: none;
  }

  .select-field__option:hover,
  .select-field__option.selected {
    background: rgba(43, 124, 114, 0.12);
    transform: none;
  }

  .select-field__check {
    color: var(--primary-strong);
    font-weight: 700;
  }

  .select-field__empty {
    padding: 0.9rem;
    color: var(--muted);
    text-align: center;
    font-size: 0.9rem;
  }
</style>
