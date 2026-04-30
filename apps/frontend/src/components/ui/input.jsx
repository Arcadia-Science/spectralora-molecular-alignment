import { cn } from '@/lib/utils'

export function Input({ className, ...props }) {
  return (
    <input
      className={cn(
        'flex h-9 w-full rounded-md border border-border bg-muted px-3 py-1 text-sm text-foreground',
        'placeholder:text-muted-foreground',
        'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary',
        'disabled:cursor-not-allowed disabled:opacity-50',
        'font-mono',
        className
      )}
      {...props}
    />
  )
}
