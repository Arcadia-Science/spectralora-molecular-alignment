import * as LabelPrimitive from '@radix-ui/react-label'
import { cn } from '@/lib/utils'

export function Label({ className, ...props }) {
  return (
    <LabelPrimitive.Root
      className={cn(
        'text-xs font-medium leading-none text-muted-foreground',
        'peer-disabled:cursor-not-allowed peer-disabled:opacity-70',
        className
      )}
      {...props}
    />
  )
}
