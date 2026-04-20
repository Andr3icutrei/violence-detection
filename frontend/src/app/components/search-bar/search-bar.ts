import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { TranslatePipe } from '@ngx-translate/core';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { required } from '@angular/forms/signals';
import { debounceTime, distinctUntilChanged, of, switchMap } from 'rxjs';

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
        })
      ).subscribe();
  }
}
