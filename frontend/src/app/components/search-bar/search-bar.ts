import { Component, DestroyRef, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { debounceTime, distinctUntilChanged, of, switchMap } from 'rxjs';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

@Component({
  selector: 'app-search-bar',
  imports: [ReactiveFormsModule],
  templateUrl: './search-bar.html',
  styleUrl: './search-bar.css',
})
export class SearchBar implements OnInit {
  searchControl!: FormControl;
  @Input({required : true}) placeholderText!: string;
  @Output() onDebouncedSearch: EventEmitter<string> = new EventEmitter();
  private currentSearchTerm: string = '';

  constructor(private destroyRef: DestroyRef) {}

  ngOnInit(): void {
    this.searchControl = new FormControl('', { updateOn: 'change' });
    this.searchControl.valueChanges
      .pipe(
        debounceTime(300),
        distinctUntilChanged(),
        switchMap((searchTerm) => {
          this.currentSearchTerm = searchTerm || '';
          this.onDebouncedSearch.emit(this.currentSearchTerm);
          return of([]);
        }),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe();
  }
}
